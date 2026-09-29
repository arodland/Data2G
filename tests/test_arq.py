"""ARQ link core: codecs, and two stations in lockstep through a fake
PHY with loss, asserting the accounting (docs/arq.md §10)."""

import random
from collections import Counter

import pytest

from data2g.arq import frames as F
from data2g.arq import link as L


# --- codecs -------------------------------------------------------------------------

def test_core_roundtrip():
    rng = random.Random(0)
    for _ in range(500):
        c = F.Core(ftype=rng.randrange(4), n_ctl=rng.randint(1, 4), burst_seq=rng.randrange(8),
                   acted_on=rng.randrange(8), cum=rng.randrange(128), reply_lost=rng.random() < 0.5,
                   k=rng.randrange(64), recommend=rng.randrange(64), size_hint=rng.randrange(4))
        assert F.Core.unpack(c.pack()) == c


def test_control_roundtrip_across_codewords():
    rng = random.Random(1)
    for pb in (4, 10, 22, 46):
        for _ in range(100):
            ext = {t: bytes(rng.randrange(256) for _ in range(rng.randint(0, 3))) for t in (F.T_NEW, F.T_RV)}
            ctl = F.Control(F.Core(k=3), ext)
            try:
                payloads = ctl.pack(pb)
            except ValueError:
                continue
            assert all(len(p) == pb for p in payloads)
            back = F.Control.unpack(payloads)
            assert back.core.k == 3 and back.core.n_ctl == len(payloads) and back.ext == ext


def test_bitmap_rv_callsign_records():
    for cum in (0, 5, 100, 127):
        got = {(cum + d) % 128 for d in (1, 3, 64)}
        assert F.unpack_bitmap(F.pack_bitmap(got, cum), cum) == got
    assert F.pack_bitmap(set(), 7) == b""
    rvs = [0, 1, 2, 3, 3, 2, 1]
    assert F.unpack_rv(F.pack_rv(rvs), len(rvs)) == rvs
    for call in ("W1AW", "VK2ABC-15", "G4ABC/P", "KD9XYZ-1"):
        assert F.unpack_call(F.pack_call(call)) == call
    r = F.RecordReader()
    data = bytes(range(256)) * 3
    stream = F.to_records(data[:300]) + bytes(5) + F.to_records(data[300:])
    out = b"".join(r.feed(stream[i:i + 7]) for i in range(0, len(stream), 7))
    assert out == data


# --- lockstep harness -------------------------------------------------------------

MODES = {"m4": (4, 1), "m22": (22, 4), "m46": (46, 4)}  # name -> (payload bytes, rv cycle)
# CPM-like: control in its own short codeword, one per burst (twice when
# duplicated), at most 8 data codewords
CPM_LIKE = {"c60": (60, 4), "c40": (40, 4)}
MODES.update(CPM_LIKE)
CTL_BYTES = 20


class RandomPolicy:
    def __init__(self, rng, change=0.2, modes=("m4", "m22", "m46"), max_cw=20):
        self.rng, self.change, self.modes, self.max_cw = rng, change, modes, max_cw
        self.dup_rng = random.Random(max_cw * 1000 + int(change * 100))
        self.mode = rng.choice(modes)

    def choose(self, station, escalation):
        if escalation:  # fewest control codewords, short burst (what a shifter would do)
            return max(self.modes, key=lambda m: MODES[m][0]), 2
        if self.rng.random() < self.change:
            self.mode = self.rng.choice(self.modes)
        if self.mode in CPM_LIKE:
            return self.mode, self.rng.randint(1, min(self.max_cw, 10))
        return self.mode, self.rng.randint(1, self.max_cw)

    def payload_bytes(self, m):
        return MODES[m][0]

    def ctl_payload_bytes(self, m):
        return CTL_BYTES if m in CPM_LIKE else MODES[m][0]

    def max_ctl(self, m):
        return 1 if m in CPM_LIKE else 4

    @property
    def want_dup(self):
        """Duplicated control (ARQ_DUP) asked for at random: its layout
        must keep every accounting rule. Its own random stream, so the
        loss pattern stays the one each seed had without it."""
        return self.dup_rng.random() < 0.3

    def rv_cycle(self, m):
        return MODES[m][1]


class FakeRx:
    """What a receiver gets from one TxBurst: some slots lost; soft-bit
    combining modelled per key. Counts requests whose (mask, rv) differ
    from what was sent: the accounting disagreements."""

    def __init__(self, burst, rng, p_cw, store, stats):
        self.burst, self.store, self.stats = burst, store, stats
        self.submode, self.n_cw = burst.submode, len(burst.slots)
        self.good = [rng.random() >= p_cw for _ in burst.slots]
        self.rng = rng

    def decode(self, i, mask_id, rv, key):
        s = self.burst.slots[i]
        if mask_id[2] >= F.SEQ_MOD:  # control: fresh, never combined
            return s.payload if (mask_id == s.mask_id and self.good[i]) else None
        if mask_id != s.mask_id or rv != s.rv:
            self.stats["mismatch"] += 1
            return None
        if self.good[i]:
            return s.payload
        held = self.store.setdefault(key, [])
        if any(h != (self.submode, s.payload) for h in held):
            # soft bits of another codeword under this key (a re-sliced seq):
            # combining them is an accounting bug even when the CRC saves it
            self.stats["mismatch"] += 1
        held.append((self.submode, s.payload))
        if len(held) >= 2 and self.rng.random() < 0.6:
            return s.payload  # soft combining pulled it through
        return None

    def forget(self, key):
        self.store.pop(key, None)


LAST_REASON = [""]
WORDS = (b"the of and to in is that for with as burst codeword ACK resend CQ de QTH RST 599 73 "
         b"frequency antenna propagation Winlink message\r\n").split(b" ")


def text(rng, n):
    """Compressible bytes, not a pattern deflate learns in one codeword."""
    out = bytearray()
    while len(out) < n:
        out += rng.choice(WORDS) + b" " + (str(rng.randrange(1000)).encode() if rng.random() < 0.1 else b"")
    return bytes(out[:n])


def payload(rng, n, kind):
    """kind "random", "text", or "mixed": text and random runs of 50-600 B
    alternating, so compressed and raw codewords interleave."""
    noise = lambda k: bytes(rng.randrange(256) for _ in range(k))
    if kind != "mixed":
        return text(rng, n) if kind == "text" else noise(n)
    out = bytearray()
    while len(out) < n:
        k = rng.randrange(50, 600)
        out += text(rng, k) if rng.random() < 0.5 else noise(k)
    return bytes(out[:n])


def count_resends(burst, sender, stats):
    """Resends in `burst`: raw, or compressed with the T_COMP bit sent, or
    omitted (the peer holds it from the burst that sent the codeword new)."""
    n_ctl = sum(s.mask_id[2] >= F.SEQ_MOD for s in burst.slots)
    c = F.Control.unpack([s.payload for s in burst.slots[:n_ctl] if s.rv == 0])
    bits = F.unpack_flags(c.ext.get(F.T_COMP, b""), c.core.k)
    for s, bit in zip(burst.slots[n_ctl:n_ctl + c.core.k], bits):
        cw = next((x for x in sender.tx.cws.values() if x.seq % F.SEQ_MOD == s.mask_id[2]), None)
        if cw is not None:
            stats["resend_raw" if not cw.comp else "resend_comp_bit" if bit else "resend_comp_implicit"] += 1


def run(seed, p_burst, p_cw, n_a, n_b, max_turns=4000, die_at=None, change=0.2, max_cw=20,
        modes=("m4", "m22", "m46"), kind="random"):
    rng = random.Random(seed)
    data_a = payload(rng, n_a, kind)
    data_b = payload(rng, n_b, kind)
    a = L.Station(0, RandomPolicy(random.Random(seed + 1), change, modes, max_cw), master=True)
    b = L.Station(1, RandomPolicy(random.Random(seed + 2), change, modes, max_cw))
    a.write(data_a)
    b.write(data_b)
    stores = {0: {}, 1: {}}
    stats = Counter(mismatch=0, turns=0)
    got_a, got_b = bytearray(), bytearray()
    LAST_REASON[0] = ""
    burst, sender = a.build(), a
    for turn in range(max_turns):
        stats["turns"] = turn
        count_resends(burst, sender, stats)
        receiver = b if sender is a else a
        dead = die_at is not None and turn >= die_at
        ok = False
        if not dead and rng.random() >= p_burst:
            ok = receiver.handle(FakeRx(burst, rng, p_cw, stores[receiver.direction], stats))
        got_a += a.read()
        got_b += b.read()
        # never a wrong byte, at any point
        assert bytes(got_b) == data_a[:len(got_b)]
        assert bytes(got_a) == data_b[:len(got_a)]
        if a.state == L.FAILED or b.state == L.FAILED:
            LAST_REASON[0] = a.fail_reason or b.fail_reason
            return "failed", stats
        if got_a == data_b and got_b == data_a and not a.tx.pending() and not b.tx.pending():
            # the log's throughput counters: exact, abandons and resends included
            assert (a.tx.acked, b.rx.reader.delivered) == (len(F.to_records(data_a)), n_a)
            assert (b.tx.acked, a.rx.reader.delivered) == (len(F.to_records(data_b)), n_b)
            for k in ("cw_new", "cw_comp"):
                stats[k] = a.stats[k] + b.stats[k]
            return "done", stats
        if ok:
            burst, sender = receiver.build(), receiver
            receiver.answered()
        else:
            burst, sender = a.on_timeout(), a
            if burst is None:
                LAST_REASON[0] = a.fail_reason
                return "failed", stats
    return "stuck", stats


@pytest.mark.parametrize("seed", range(40))
def test_clean_and_lossy_links_deliver_exactly(seed):
    p_burst = [0.0, 0.1, 0.2][seed % 3]
    p_cw = [0.0, 0.1, 0.3][seed // 3 % 3]
    result, stats = run(seed, p_burst, p_cw, n_a=random.Random(seed).randrange(0, 4000),
                        n_b=random.Random(seed + 99).randrange(0, 1500))
    if (p_burst, p_cw) == (0.2, 0.3):
        # the heaviest cell: ~20% of runs lose the link (bounded, and never a
        # wrong byte), with duplicated control asked or not (300 seeds: 59
        # without, 48 with); which seeds do is luck
        assert result == "done" or (result == "failed" and LAST_REASON[0] == "link lost"), (result, stats)
    else:
        assert result == "done", (result, stats)
    assert stats["mismatch"] == 0


@pytest.mark.parametrize("seed", range(30))
def test_short_control_codewords_mixed_with_ofdm(seed):
    """CPM modes carry control in a 20 B codeword, one per burst: control
    sheds optional extensions and the bitmap to fit, and mode changes
    between the families keep every accounting rule."""
    p_burst = [0.0, 0.1, 0.2][seed % 3]
    p_cw = [0.0, 0.1, 0.2][seed // 3 % 3]
    result, stats = run(400 + seed, p_burst, p_cw, random.Random(seed).randrange(0, 3000),
                        random.Random(seed + 99).randrange(0, 1500), modes=("m4", "m46", "c60", "c40"))
    assert result == "done" or (p_burst, p_cw) == (0.2, 0.2) and LAST_REASON[0] == "link lost", (result, stats)
    assert stats["mismatch"] == 0


@pytest.mark.parametrize("seed", range(20))
def test_full_window_bursts(seed):
    """64-codeword bursts fill the send window: a reply acking all of it
    puts the cumulative exactly WINDOW past the base, which must not alias
    in 7 bits (linksim found WINDOW = SEQ_MOD / 2 failing the link here)."""
    result, stats = run(300 + seed, [0.0, 0.2][seed % 2], [0.0, 0.05, 0.2][seed % 3], 30000, 3000,
                        change=0.05, max_cw=64)
    assert result == "done", (result, stats, LAST_REASON[0])
    assert stats["mismatch"] == 0


@pytest.mark.parametrize("seed", range(10))
def test_heavy_loss_ends_bounded(seed):
    result, stats = run(100 + seed, 0.5, 0.5, 3000, 3000, max_turns=20000)
    assert result in ("done", "failed")
    assert stats["mismatch"] == 0


@pytest.mark.parametrize("seed", range(5))
def test_dead_link_fails_within_bound(seed):
    result, stats = run(200 + seed, 0.0, 0.0, 20000, 0, die_at=10)
    assert result == "failed"
    assert stats["turns"] <= 10 + L.LINK_LOST_MISSES + 1


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("max_cw", (2, 3))
def test_tiny_bursts_with_losses_still_deliver(seed, max_cw):
    """Bursts of 1-3 slots, a fifth of them lost, duplicated control asked
    for at random: control can crowd out data in a burst, but never for
    good, and every byte arrives. Written for the sender-side hedges after
    a miss (README: measured and dropped); it guards the tight-burst path
    whatever claims its slots next."""
    result, stats = run(300 + seed, 0.2, 0.05, 1500, 1500, max_cw=max_cw, modes=("m22",))
    assert result == "done", (result, LAST_REASON[0])
    assert stats["mismatch"] == 0


@pytest.mark.parametrize("seed", range(30))
def test_compressed_codewords_deliver_exactly(seed):
    """Text goes as deflate primed with the delivered stream (T_COMP):
    through losses, abandons, re-slicing, CPM's 20 B control and tiny
    bursts, every byte arrives exact and compression is used."""
    modes = [("m22", "m46"), ("m4", "m46", "c60", "c40"), ("m22",)][seed % 3]
    p_burst, p_cw = [(0.0, 0.0), (0.1, 0.1), (0.2, 0.05)][seed // 3 % 3]
    result, stats = run(500 + seed, p_burst, p_cw, 6000, 2000, modes=modes, max_cw=[20, 10, 3][seed % 3],
                        kind="text")
    assert result == "done", (result, stats, LAST_REASON[0])
    assert stats["mismatch"] == 0
    assert stats["cw_comp"] > 0


@pytest.mark.parametrize("seed", range(20))
def test_mixed_compressed_and_raw_through_retransmits(seed):
    """Text and random runs interleaved: compressed and raw codewords in
    the same bursts, both resent (with RVs, after abandons) and both
    delivered exact. A raw resend must keep its T_COMP bit clear and a
    compressed one set, or the stream corrupts."""
    modes = [("m22", "m46"), ("m4", "m46", "c60", "c40")][seed % 2]
    result, stats = run(700 + seed, 0.0, 0.2, 8000, 3000, modes=modes, max_cw=[20, 10][seed % 2], kind="mixed")
    assert result == "done", (result, stats, LAST_REASON[0])
    assert stats["mismatch"] == 0
    assert 0 < stats["cw_comp"] < stats["cw_new"]
    assert stats["resend_comp_implicit"] > 0 and stats["resend_raw"] > 0, stats


def test_compressed_resend_after_lost_control_carries_its_bit():
    """A compressed codeword whose first burst the peer never decoded goes
    again with its T_COMP bit sent: the peer can't know it. Whole bursts
    lost, so that path runs; bursts deliver exactly or the link is lost."""
    total = Counter()
    for seed in range(10):
        result, stats = run(800 + seed, 0.2, 0.2, 6000, 2000, modes=("m22", "m46"), kind="mixed")
        assert result == "done" or LAST_REASON[0] == "link lost", (result, LAST_REASON[0])
        assert stats["mismatch"] == 0
        total += stats
    assert total["resend_comp_bit"] > 0 and total["resend_comp_implicit"] > 0, total


def test_compression_fits_more_per_codeword():
    """Clean link, one 46 B mode: text takes well under the raw codeword count."""
    n = 20000
    result, stats = run(1, 0.0, 0.0, n, 0, change=0.0, modes=("m46",), kind="text")
    assert result == "done"
    raw = -(-len(F.to_records(text(random.Random(1), n))) // 46)
    assert stats["cw_new"] < raw / 1.5, (stats["cw_new"], raw)


def test_compression_off_sends_raw_and_still_receives(monkeypatch):
    """DATA2G_COMPRESS=0 (link.COMPRESS): a sender sends raw; its peer still
    compresses, and each side takes what the other sends."""
    rng = random.Random(3)
    a = L.Station(0, RandomPolicy(random.Random(1), 0.0, ("m46",), 20), master=True)
    b = L.Station(1, RandomPolicy(random.Random(2), 0.0, ("m46",), 20))
    up, down = text(rng, 3000), text(rng, 3000)
    a.write(up)
    b.write(down)
    got_a, got_b, burst, sender = bytearray(), bytearray(), None, b
    for _ in range(200):
        receiver = b if sender is a else a
        monkeypatch.setattr(L, "COMPRESS", receiver is b)  # a never compresses
        burst = receiver.build()
        receiver.answered()
        (a if receiver is b else b).handle(FakeRx(burst, rng, 0.0, {}, Counter()))
        sender = receiver
        got_a += a.read()
        got_b += b.read()
        if got_b == up and got_a == down:
            break
    assert bytes(got_b) == up and bytes(got_a) == down
    assert a.stats["cw_comp"] == 0 and b.stats["cw_comp"] > 0


def test_incompressible_goes_raw():
    result, stats = run(2, 0.0, 0.0, 3000, 3000, modes=("m46",))
    assert result == "done" and stats["cw_new"] > 0 and stats["cw_comp"] == 0


def test_escalation_floor_is_sticky_and_decays():
    """A drop that took escalation 4 to recover makes the next drop start
    at 4 (no identical repeat, no climb); clean turns lower it again."""
    class Rec(RandomPolicy):
        def choose(self, station, escalation):
            station.seen = getattr(station, "seen", []) + [escalation]
            return super().choose(station, escalation)

    rng = random.Random(5)
    a = L.Station(0, Rec(random.Random(1), 0.0, ("m46",), 4), master=True)
    b = L.Station(1, Rec(random.Random(2), 0.0, ("m46",), 4))
    a.write(text(rng, 4000))

    def hear(dst, burst):
        assert dst.handle(FakeRx(burst, rng, 0.0, {}, Counter()))

    def turn():
        hear(b, a.build())
        hear(a, b.build())

    turn()
    hear(b, a.build())
    b.build()  # its reply is lost
    assert a.on_timeout() is a.last_sent  # the identical repeat, lost
    a.seen = []
    for _ in range(3):  # polls at 2, 3, 4: the last one heard and answered
        p = a.on_timeout()
    assert a.seen == [2, 3, 4]
    hear(b, p)
    hear(a, b.build())
    assert a.esc_floor == 4
    turn()
    assert b.esc_floor >= 1

    hear(b, a.build())
    b.build()  # lost again
    a.seen = []
    p = a.on_timeout()
    assert a.seen == [4]  # built, not the identical repeat: straight to 4
    hear(b, p)
    hear(a, b.build())
    for _ in range(L.FLOOR_DECAY_TURNS):
        turn()
    assert a.esc_floor == 3
