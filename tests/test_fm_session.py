"""FM ARQ sessions end to end through the FM simulator (FM profile): the real
modem, ARQ and gear shifter (scripts/phy_session.py's RealPhy on
FmSessionChannel), each station's TX path matching its radio. At a
comfortable CNR the shifter must climb onto FM's own ladders (f55 narrow,
f108 wide) and deliver every byte."""

import sys
from pathlib import Path

import pytest

from data2g import config, modem

pytestmark = pytest.mark.skipif(config.PROFILE != "fm", reason="FM profile only")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


@pytest.mark.parametrize("radio, cap, band", [("mic", 1, "f55"), ("flat", 2, "f108")])
def test_fm_session_climbs_onto_its_ladder(monkeypatch, radio, cap, band):
    import linksim as L
    import phy_session as PS

    monkeypatch.setattr(modem, "TX_PATH", radio)
    ch = PS.FmSessionChannel("static", radio, 30.0, seed=3, horizon=90.0)
    phy = PS.RealPhy(ch)
    sent = []
    hear = phy.hear

    def spy(x, t0, rx=None):
        r = hear(x, t0, rx)
        if r is not None and r.get("spec") is not None:
            sent.append(r["spec"].name)
        return r

    phy.hear = spy
    steps = [("a", 30_000, None, 0.0)]
    res = L.run(L.make_policy("shift"), L.make_policy("shift"), None, steps, seed=3, horizon=90.0, phy=phy, cap=cap)
    assert res["delivered"] >= 30_000, res
    assert any(n.startswith(band + "-") for n in sent), sent
    assert not any(n.startswith(("n4-", "w48-")) for n in sent), sent
