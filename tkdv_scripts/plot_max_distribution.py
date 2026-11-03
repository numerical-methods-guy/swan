import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kdv.tkdv_solver import TKdVSolver

# --- Config ---
J        = 32
E0       = 100.0
L0       = 6.0
N_TRAJ   = 1000
BURNIN   = 1000
THIN     = 10

CASES = [
    ("upstream",   1.0,  -0.5,  0.01, 200),
    ("downstream", 0.24, -0.32, 0.01,  50),
]

OUT_PNG = os.path.join(os.path.dirname(__file__), "max_distribution.png")


def max_amplitudes(label, D, theta, dt, N):
    print(f"\n=== {label.upper()} (D={D}, θ={theta}, N_traj={N_TRAJ}) ===")
    solver = TKdVSolver(J=J, dt=dt, D=D, E0=E0, L0=L0)

    # Sample N_TRAJ ICs from one long MCMC chain
    print("Sampling Gibbs ICs ...")
    u_hat = solver.gibbs_initial_condition(
        theta=theta, n_burnin=BURNIN, n_samples=N_TRAJ, thin=THIN
    )  # (N_TRAJ, Lambda)

    # Run all trajectories in one batched forward pass
    print(f"Running {N} steps x {N_TRAJ} trajectories (batched) ...")
    max_vals = torch.zeros(N_TRAJ, dtype=torch.float64)

    with torch.no_grad():
        u_hat_cur = u_hat.clone()
        for step in range(N + 1):
            u_phys = solver.spec2phys(u_hat_cur)          # (N_TRAJ, J)
            step_max = u_phys.abs().max(dim=-1).values     # (N_TRAJ,)
            # propagate NaN for blown-up trajectories
            max_vals = torch.where(
                torch.isnan(step_max), torch.full_like(max_vals, float("nan")),
                torch.maximum(max_vals, step_max)
            )
            if step < N:
                u_hat_cur = solver.timestep(u_hat_cur, nsteps=1)

    vals = max_vals.numpy()
    n_nan = np.isnan(vals).sum()
    vals = vals[~np.isnan(vals)]
    if n_nan > 0:
        print(f"  (dropped {n_nan} blow-up trajectories, {len(vals)} remain)")
    print(f"Max amplitude stats: mean={vals.mean():.3f}, "
          f"p95={np.percentile(vals, 95):.3f}, max={vals.max():.3f}")
    return vals


results = {}
for label, D, theta, dt, N in CASES:
    results[label] = max_amplitudes(label, D, theta, dt, N)

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

colors = {"upstream": "steelblue", "downstream": "crimson"}
for ax, (label, vals) in zip(axes, results.items()):
    ax.hist(vals, bins=50, color=colors[label], alpha=0.8, edgecolor="white", lw=0.3)
    ax.axvline(np.percentile(vals, 95), color="k", ls="--", lw=1, label="p95")
    ax.axvline(np.percentile(vals, 99), color="k", ls=":",  lw=1, label="p99")
    ax.set_xlabel("max |u(x,t)|")
    ax.set_ylabel("count")
    D_val = dict(zip([c[0] for c in CASES], [c[1] for c in CASES]))[label]
    theta_val = dict(zip([c[0] for c in CASES], [c[2] for c in CASES]))[label]
    ax.set_title(f"{label}  [D={D_val}, θ={theta_val}]")
    ax.legend()

fig.suptitle("Distribution of max amplitude over 1000 trajectories", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"\nSaved to {OUT_PNG}")
