"""Event-harness half-duplex ordering; no DSP or probabilistic channel."""
import importlib.util
from pathlib import Path
import random

import pytest

from data2g.arq import session as S
from test_arq import FakeRx
from test_arq_session import Policy


def load_linksim():
    spec = importlib.util.spec_from_file_location('study_linksim', Path(__file__).parents[1] / 'scripts/linksim.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class LongPolicy(Policy):
    def choose(self, station, escalation):
        return 'm46', 3

    def airtime(self, mode, n_cw, dup=False):
        return 5.0


class HalfDuplexPhy:
    def __init__(self):
        self.air = []
        self.sessions = {}

    def send(self, burst, start):
        sender = burst.slots[0].mask_id[1] & 1
        end = start + 5.0
        assert not any(who == sender and tail > start for _, tail, who in self.air), 'station keyed over its own transmission'
        self.air.append((start, end, sender))
        return end, (start + .5, burst.submode, len(burst.slots)), lambda store, stats, rng: FakeRx(burst, rng, 0, store, stats), None


@pytest.mark.parametrize('think', [0.0, 0.2, 1.0])
def test_application_write_during_callee_reply_does_not_poll_the_transmitter(think):
    """B writes after A's data arrives, while B's long reply is on air.
    Its previous wake timer can expire before that reply ends. As Engine,
    the event loop must not poll B until its TX-end callback resets it."""
    sim = load_linksim()
    phy = HalfDuplexPhy()
    result = sim.run(LongPolicy(random.Random(1), change=0), LongPolicy(random.Random(2), change=0),
                     None, [('a', 64, None, 0), ('b', 64, 0, think)], seed=1, horizon=180, phy=phy)
    assert result['complete'] and result['delivered'] == 128
    assert result['collisions'] == result['mismatch'] == 0
    assert all(s.state == S.CONNECTED for s in phy.sessions.values())
    assert all(not s.close_reason for s in phy.sessions.values())
