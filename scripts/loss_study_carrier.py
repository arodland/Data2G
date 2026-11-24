"""scripts/loss_study.py with one more interference preset, `carrier12`: a
steady carrier at 1573.4 Hz (inside every CPM grid; tests/test_cpm.py's),
12 dB over the floor in 50 Hz, for the whole session: #295's cell.

    DATA2G_PEP_REF_DB=5 python -m scripts.loss_study_carrier --cells awgn:-12:carrier12,awgn:-12 ...
"""

from scripts import loss_study as L  # first: it sets the thread caps before numpy loads

import os  # noqa: E402

if os.environ.get("DATA2G_STUDY_ROBUST"):  # GearShifter.robust_mode() always this mode (scripts/connect_session_study.py's)
    from data2g.arq import policy as _G

    _G.GearShifter.robust_mode = lambda self, _m=os.environ["DATA2G_STUDY_ROBUST"]: _m

L.INTF.PRESETS["carrier12"] = L.INTF.Spec(qrm=(L.INTF.Qrm("carrier", 1573.4, 50, 12.0, 600.0, 1.0),))

if __name__ == "__main__":
    L.main()
