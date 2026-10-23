"""Optuna search over the shifter's decision knobs (env study toggles),
scored by paired loss studies against the shipped values on the same seeds.

    python scripts/policy_search.py --name knobs1 --trials 40
    python scripts/policy_search.py --name knobs1 --confirm 17 --seed0 144

Each trial is one loss_study.py run (tools/with_native.py; any toggle set
keeps the shifter in Python, so the baseline sets the shipped values
explicitly and takes the same path). Two objectives, both maximised:
the mean per-cell % change in delivered bps, and the worst cell's paired
t (bottom cells move 4-10% from noise alone, so a % floor would chase it).
Optuna keeps the Pareto front. The search reuses its seeds every trial, so
it fits their noise: confirm a pick on untouched seeds (--confirm).

--space FILE: a module with suggest(trial) -> {env var: value} and SHIPPED
({param: value}, the baseline); default: this file's. Resumable: the study
lives in DIR/study.db, finished trials' CSVs are reused.
"""

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import optuna

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import paired_loss  # noqa: E402

# runs/cnpd_round.sh's judge cells: fixed, the bottom, and slow fading
CELLS = ("awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8,mpd:-6,mpp:-8,awgn:15,mpp:-6,mpd:20,mpg:15,"
         "mpp:15,mpg:-4,awgn:-4,awgn:4,awgn:8,mpg:4,mpg:12,mpp:-10,mpd:-8,"
         "awgn:4:slow,awgn:15:slow,mpg:8:slow,mpp:0:slow,mpd:0:slow")

# The installed model's decision knobs, at their shipped values (predictor.LOGIT_OFFSETS,
# policy.CPM_FLOOR_DEFAULT, policy.NOISE_RULE; the qpsk-r1/5 fast-fading bias is baked
# into the distilled CNP, so cond_qpsk is on top of it).
SHIPPED = dict(off_16qam=-1.0, off_256l=-0.5, cond_qpsk=0.0, floor_snr=-4.0, floor_s=1.0, noise_rule=1.0)


def suggest(t):
    p = dict(off_16qam=t.suggest_float("off_16qam", -2.5, 0.5), off_256l=t.suggest_float("off_256l", -2.0, 0.5),
             cond_qpsk=t.suggest_float("cond_qpsk", -1.5, 1.0), floor_snr=t.suggest_float("floor_snr", -7.0, -1.0),
             floor_s=t.suggest_float("floor_s", 0.0, 2.0), noise_rule=t.suggest_float("noise_rule", 0.0, 2.0))
    return {"DATA2G_LOGIT_OFFSETS": f"w48-16qam-r1/2:{p['off_16qam']:.3f},n10-256l-r3/4:{p['off_256l']:.3f}",
            "DATA2G_COND_OFFSETS": f"0.7:-2:qpsk-r1/5:{p['cond_qpsk']:.3f}",
            "DATA2G_CPM_FLOOR": f"snr:{p['floor_snr']:.3f}:{p['floor_s']:.3f}",
            "DATA2G_NOISE_RULE": f"{p['noise_rule']:.3f}"}


def run(out: Path, env: dict, a, seed0: int):
    """One loss study with `env` on top; skipped when its CSV is already there."""
    if out.exists() and out.stat().st_size:
        return
    tmp = out.with_suffix(".part")
    caps = {v: "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    cmd = [sys.executable, str(ROOT / "tools/with_native.py")]
    cmd += ["--dd-budget", str(a.dd_budget)] if a.dd_budget is not None else []
    cmd += [str(ROOT / "scripts/loss_study.py"), "--out", str(tmp), "--seed0", str(seed0), "--seeds", str(a.seeds),
            "--horizon", str(a.horizon), "--jobs", str(a.jobs), "--policy", "shift+cpm", "--cells", a.cells]
    with open(out.with_suffix(".log"), "w") as log:
        subprocess.run(cmd, env={**os.environ, **caps, "DATA2G_PEP_REF_DB": "5", **env}, stdout=log,
                       stderr=subprocess.STDOUT, check=True, cwd=ROOT)
    tmp.rename(out)  # only a finished study counts on resume


def score(base: Path, arm: Path):
    """-> (mean per-cell % change, worst cell's paired t, {cell: (%, t)})."""
    A, B = paired_loss.delivered(base), paired_loss.delivered(arm)
    cells = {}
    for k in sorted(A):
        seeds = sorted(set(A[k]) & set(B.get(k, {})))
        x, y = np.array([A[k][s] for s in seeds]), np.array([B[k][s] for s in seeds])
        if len(seeds) < 2 or x.mean() <= 0:
            continue
        d = y - x
        se = d.std(ddof=1) / np.sqrt(len(d))
        cells[f"{k[0]}{k[1]:+.0f}"] = (100 * d.mean() / x.mean(), d.mean() / se if se > 0 else 0.0)
    pct = np.array([v[0] for v in cells.values()])
    return float(pct.mean()), float(min(v[1] for v in cells.values())), cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="study name; its files go in runs/policy_search/NAME")
    ap.add_argument("--trials", type=int, default=20, help="trials to add this invocation")
    ap.add_argument("--space", default=None, help="a module with suggest(trial) and SHIPPED (default: this file's)")
    ap.add_argument("--cells", default=CELLS)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=0, help="the search's seeds; with --confirm, the fresh ones")
    ap.add_argument("--horizon", type=float, default=600.0)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--dd-budget", type=float, default=None, help="with_native --dd-budget (inf: no wall-clock noise)")
    ap.add_argument("--confirm", type=int, default=None, metavar="TRIAL",
                    help="re-run the baseline and this trial on --seed0's seeds and print the paired table")
    a = ap.parse_args()

    space = sys.modules[__name__]
    if a.space:
        spec = importlib.util.spec_from_file_location("space", a.space)
        space = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(space)
    d = ROOT / "runs/policy_search" / a.name
    d.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name=a.name, storage=f"sqlite:///{d / 'study.db'}", load_if_exists=True,
                                directions=["maximize", "maximize"])
    base_env = space.suggest(optuna.trial.FixedTrial(space.SHIPPED))

    if a.confirm is not None:
        t = study.trials[a.confirm]
        base, arm = d / f"base_s{a.seed0}.csv", d / f"t{a.confirm}_s{a.seed0}.csv"
        run(base, base_env, a, a.seed0)
        run(arm, space.suggest(optuna.trial.FixedTrial(t.params)), a, a.seed0)
        print(f"== seeds {a.seed0}-{a.seed0 + a.seeds - 1}: shipped vs trial {a.confirm} {t.params}")
        paired_loss.main(str(base), str(arm))
        return

    base = d / f"base_s{a.seed0}.csv"
    run(base, base_env, a, a.seed0)

    def objective(trial):
        out = d / f"t{trial.number}_s{a.seed0}.csv"
        env = space.suggest(trial)
        trial.set_user_attr("env", env)
        run(out, env, a, a.seed0)
        mean, worst, cells = score(base, out)
        trial.set_user_attr("cells", cells)
        print(f"trial {trial.number}: mean {mean:+.2f}%, worst t {worst:+.2f}  {env}", flush=True)
        return mean, worst

    study.optimize(objective, n_trials=a.trials)
    print("\nPareto front (mean %, worst t):")
    for t in sorted(study.best_trials, key=lambda t: -t.values[0]):
        print(f"  trial {t.number}: {t.values[0]:+.2f}%, {t.values[1]:+.2f}  {t.params}")


if __name__ == "__main__":
    main()
