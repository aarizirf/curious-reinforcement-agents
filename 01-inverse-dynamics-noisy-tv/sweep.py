"""Sweep EMB_DIM x {FROZEN, SLOT} and print the results table.

    python sweep.py             show each epoch as it trains
    python sweep.py -q          table only
    python sweep.py --seed 1    another draw
"""

import copy
import sys
import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning

import icm
import probe
from corridor import K_BITS, N_POS
from rollout import collect

warnings.filterwarnings("ignore", category=ConvergenceWarning)
torch.set_num_threads(4)

EMB_DIMS = (2, 4, 8, 32)
N_TRANSITIONS = 100_000
N_EVAL = 20_000
MLP_BITS = 8  # bits are iid; 8 is enough for a mean and the MLP is slow
SEED = 0


def run(condition, emb_dim, seed, verbose):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    obs, act, next_obs, _ = collect(condition, N_TRANSITIONS, rng)
    phi, inv = icm.make_encoder(emb_dim), icm.Inverse(emb_dim)
    phi_init = copy.deepcopy(phi)

    def log(epoch, loss, acc):
        print(f"    epoch {epoch + 1:>2}/{icm.EPOCHS}   loss {loss:.4f}   "
              f"train acc {acc:.4f}", flush=True)

    icm.train(phi, inv, obs, act, next_obs, log=log if verbose else None)

    e_obs, e_act, e_next, _ = collect(condition, N_EVAL, rng)
    inv_all, inv_wall = icm.evaluate(phi, inv, e_obs, e_act, e_next)

    p_obs, p_act, p_next, p_tv = collect(condition, probe.N_TRAIN + probe.N_TEST, rng)
    return dict(
        inv_all=inv_all,
        inv_wall=inv_wall,
        lin_init=probe.content(phi_init, p_obs, p_tv, K_BITS),
        lin=probe.content(phi, p_obs, p_tv, K_BITS),
        mlp=probe.content(phi, p_obs, p_tv, MLP_BITS, nonlinear=True),
        change=probe.change(phi, p_obs, p_next, p_act)
        if condition == "SLOT"
        else float("nan"),
    )


def main():
    verbose = "-q" not in sys.argv
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else SEED
    cols = f"{'emb':>4} {'cond':<7} {'inv_all':>8} {'inv_wall':>9} {'ceil':>6} " \
           f"{'lin_init':>9} {'lin':>7} {'mlp':>7} {'change':>7}"
    rows = []

    for emb_dim in EMB_DIMS:
        for condition in ("FROZEN", "SLOT"):
            if verbose:
                print(f"\nemb={emb_dim} {condition}", flush=True)
            r = run(condition, emb_dim, seed, verbose)
            rows.append(
                f"{emb_dim:>4} {condition:<7} {r['inv_all']:>8.4f} "
                f"{r['inv_wall']:>9.4f} "
                f"{'~0.50' if condition == 'FROZEN' else '~1.00':>6} "
                f"{r['lin_init']:>9.4f} {r['lin']:>7.4f} {r['mlp']:>7.4f} "
                f"{r['change']:>7.4f}"
            )

    print(f"\n\n{K_BITS} TV bits of noise vs {N_POS} positions (3 bits of signal)")
    print(f"chance = 0.5000 on every probe column\n")
    print(cols)
    print("-" * len(cols))
    for i, row in enumerate(rows):
        print(row)
        if i % 2 == 1:
            print()


if __name__ == "__main__":
    main()
