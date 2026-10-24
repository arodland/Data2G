"""Optuna search over the shifter's decision knobs (env study toggles),
scored by paired loss studies against the shipped values on the same seeds.

    python scripts/policy_search.py --name knobs3 --trials 120 --jobs 8 --parallel 2
    python scripts/policy_search.py --name knobs3 --confirm 17 --seed0 144

Each trial runs loss_study.py (tools/with_native.py; any toggle set keeps
the shifter in Python, so the baseline sets the shipped values explicitly
and takes the same path) one seed at a time, every cell, and reports the
mean per-cell % change in delivered bps so far as that seed's step; a
median pruner stops trials behind the others after --warmup seeds. The
objective is that mean over all seeds. The worst cell's upper bound, %
change + 2 SE, is a constraint (at least --floor): a regression must be
material and beyond noise (bottom cells move 4-10% from noise alone, so a %
floor would chase it; near-deterministic AWGN cells gave -0.4% a paired t
of -2.8, so a t floor chased those, runs/policy_search/knobs1). Optuna
prunes single-objective studies only, hence the constraint. The search
reuses its seeds every trial, so it fits their noise: confirm a pick on
untouched seeds (--confirm).

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
# policy.CPM_FLOOR_DEFAULT, .NOISE_RULE, .DUP_BELOW, .BIAS_STEP_DOWN/_UP; the
# qpsk-r1/5 fast-fading bias is baked into the distilled CNP, so cond_qpsk is on top of it).
SHIPPED = dict(off_16qam=-1.0, off_256l=-0.5, cond_qpsk=0.0, floor_snr=-4.0, floor_s=1.0, noise_rule=1.0,
               dup_below=0.9, step_down=1.0, step_up=1.0)


def suggest(t):
    p = dict(off_16qam=t.suggest_float("off_16qam", -2.5, 0.5), off_256l=t.suggest_float("off_256l", -2.0, 0.5),
             cond_qpsk=t.suggest_float("cond_qpsk", -1.5, 1.0), floor_snr=t.suggest_float("floor_snr", -7.0, -1.0),
             floor_s=t.suggest_float("floor_s", 0.0, 2.0), noise_rule=t.suggest_float("noise_rule", 0.0, 2.0),
             dup_below=t.suggest_float("dup_below", 0.5, 1.0), step_down=t.suggest_float("step_down", 0.0, 2.5),
             step_up=t.suggest_float("step_up", 0.0, 2.5))
    return {"DATA2G_LOGIT_OFFSETS": f"w48-16qam-r1/2:{p['off_16qam']:.3f},n10-256l-r3/4:{p['off_256l']:.3f}",
            "DATA2G_COND_OFFSETS": f"0.7:-2:qpsk-r1/5:{p['cond_qpsk']:.3f}",
            "DATA2G_CPM_FLOOR": f"snr:{p['floor_snr']:.3f}:{p['floor_s']:.3f}",
            "DATA2G_NOISE_RULE": f"{p['noise_rule']:.3f}", "DATA2G_DUP_BELOW": f"{p['dup_below']:.3f}",
            "DATA2G_BIAS_STEP_DOWN": f"{p['step_down']:.3f}", "DATA2G_BIAS_STEP_UP": f"{p['step_up']:.3f}"}


def run(out: Path, env: dict, a, seed0: int, seeds: int | None = None):
    """One loss study with `env` on top; skipped when its CSV is already there."""
    if out.exists() and out.stat().st_size:
        return
    out.parent.mkdir(exist_ok=True)
    tmp = out.with_suffix(".part")
    caps = {v: "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    cmd = [sys.executable, str(ROOT / "tools/with_native.py")]
    cmd += ["--dd-budget", str(a.dd_budget)] if a.dd_budget is not None else []
    cmd += [str(ROOT / "scripts/loss_study.py"), "--out", str(tmp), "--seed0", str(seed0), "--seeds", str(seeds or a.seeds),
            "--horizon", str(a.horizon), "--jobs", str(a.jobs), "--policy", "shift+cpm", "--cells", a.cells]
    with open(out.with_suffix(".log"), "w") as log:
        subprocess.run(cmd, env={**os.environ, **caps, "DATA2G_PEP_REF_DB": "5", **env}, stdout=log,
                       stderr=subprocess.STDOUT, check=True, cwd=ROOT)
    tmp.rename(out)  # only a finished study counts on resume


def delivered(paths):
    """paired_loss.delivered over several CSVs (a trial's per-seed studies)."""
    out = {}
    for p in paths:
        for k, v in paired_loss.delivered(p).items():
            out.setdefault(k, {}).update(v)
    return out


def score(base: Path, arm):
    """`arm`: a CSV or a list of them, on the baseline's seeds or some of them.
    -> (mean per-cell % change, worst cell's % + 2 SE (nan under 2 seeds),
    {cell: (%, paired t, % + 2 SE)})."""
    A, B = paired_loss.delivered(base), delivered(arm if isinstance(arm, list) else [arm])
    cells = {}
    for k in sorted(A):
        seeds = sorted(set(A[k]) & set(B.get(k, {})))
        x, y = np.array([A[k][s] for s in seeds]), np.array([B[k][s] for s in seeds])
        if not seeds or x.mean() <= 0:
            continue
        d = y - x
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        cells[f"{k[0]}{k[1]:+.0f}"] = (100 * d.mean() / x.mean(), d.mean() / se if se > 0 else 0.0,
                                       100 * (d.mean() + 2 * se) / x.mean())
    pct = np.array([v[0] for v in cells.values()])
    return float(pct.mean()), float(min(v[2] for v in cells.values())), cells


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
    ap.add_argument("--warmup", type=int, default=4, help="seeds a trial runs before it can be pruned")
    ap.add_argument("--floor", type=float, default=-1.0, help="the worst cell's %% + 2 SE a feasible trial keeps")
    ap.add_argument("--parallel", type=int, default=1, help="trials at once (each with --jobs workers)")
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
                                direction="maximize",
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=a.warmup - 1))
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
        env = space.suggest(trial)
        trial.set_user_attr("env", env)
        outs = []
        for i, seed in enumerate(range(a.seed0, a.seed0 + a.seeds)):
            outs.append(d / f"t{trial.number}" / f"s{seed}.csv")
            run(outs[-1], env, a, seed, 1)
            mean, worst, cells = score(base, outs)
            trial.report(mean, i)
            if trial.should_prune():
                print(f"trial {trial.number}: pruned after {i + 1} seeds at {mean:+.2f}%", flush=True)
                raise optuna.TrialPruned()
        trial.set_user_attr("cells", cells)
        trial.set_user_attr("worst", worst)
        trial.set_constraint("worst_cell", a.floor - worst)
        print(f"trial {trial.number}: mean {mean:+.2f}%, worst cell + 2 SE {worst:+.2f}%  {env}", flush=True)
        return mean

    study.optimize(objective, n_trials=a.trials, n_jobs=a.parallel)
    done = [t for t in study.trials if t.state.name == "COMPLETE"]
    print(f"\n{len(done)} complete, {len(study.trials) - len(done)} pruned or failed; "
          f"best feasible (worst cell + 2 SE >= {a.floor}%):")
    ok = [t for t in done if all(v <= 0 for v in t.constraints.values())]
    for t in sorted(ok, key=lambda t: -t.value)[:5]:
        print(f"  trial {t.number}: {t.value:+.2f}%, worst {t.user_attrs['worst']:+.2f}%  {t.params}")


if __name__ == "__main__":
    main()
