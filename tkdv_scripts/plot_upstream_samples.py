import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kdv.tkdv_solver import TKdVSolver

J   = 32
E0  = 100.0
L0  = 6.0
D   = 1.0

# dataset base dir
BASE = "/fs/hestia_Heccc/rpnatm/avg000/work/tkdv_datasets/samples"
N_SEEDS    = 2   # seeds to plot (seed_0, seed_1)
N_TRAJ     = 2   # trajectories per seed
N_STEPS    = 40  # steps per trajectory (0..40)
SUBFOLDER  = "tkdv_gibbs_dt1.0_dtsolver0.01_20261115"

out_png = os.path.join(os.path.dirname(__file__), "upstream_samples.png")

solver = TKdVSolver(J=J, dt=0.01, D=D, E0=E0, L0=L0).double()
x_arr  = solver.x.numpy()
t_arr  = np.arange(N_STEPS + 1, dtype=float)  # model timesteps (dt=1 each)

fig, axes = plt.subplots(
    N_SEEDS * N_TRAJ, 1,
    figsize=(14, 4 * N_SEEDS * N_TRAJ),
    squeeze=False,
)

row = 0
for seed in range(N_SEEDS):
    folder = os.path.join(BASE, f"seed_{seed}", SUBFOLDER)
    for traj in range(N_TRAJ):
        frames = []
        for step in range(N_STEPS + 1):
            path = os.path.join(folder, f"{traj}_{step}.pt")
            u_hat = torch.load(path, map_location="cpu")
            u = solver.spec2phys(u_hat).numpy()
            frames.append(u)
        U = np.stack(frames, axis=0)  # (41, 32)

        vmax = np.abs(U).max()
        ax = axes[row, 0]
        pcm = ax.contourf(t_arr, x_arr, U.T, levels=50, cmap="RdYlBu_r",
                          vmin=-vmax, vmax=vmax)
        ax.set_ylabel("x")
        ax.set_title(f"Upstream D={D}, seed={seed}, traj={traj}")
        fig.colorbar(pcm, ax=ax)
        row += 1

axes[-1, 0].set_xlabel("model time step")
plt.tight_layout()
plt.savefig(out_png, dpi=150)
print(f"Saved to {out_png}")
