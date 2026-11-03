import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kdv.tkdv_solver import TKdVSolver

# --- Cases: (label, D, theta, dt_snapshot, nsteps_per_snapshot, N, out_filename) ---
# dt_snapshot: time between saved frames; nsteps_per_snapshot: solver sub-steps per frame
CASES = [
    ("upstream",   1.0,  -0.5,  0.01, 1,  200, "trajectory_upstream.png"),
    ("downstream", 0.24, -0.5, 0.01, 1,  50, "trajectory_downstream.png"),
]

J      = 32
E0     = 100.0
L0     = 6.0
burnin = 1000
thin   = 10


def run_case(label, D, theta, dt_snapshot, nsteps, N, out_fname):
    dt_solver = dt_snapshot / nsteps
    out_png = os.path.join(os.path.dirname(__file__), out_fname)
    print(f"\n=== {label.upper()} (D={D}, θ={theta}, dt_snapshot={dt_snapshot}, nsteps={nsteps}) ===")

    solver = TKdVSolver(J=J, dt=dt_solver, D=D, E0=E0, L0=L0)

    print("Sampling Gibbs IC ...")
    with torch.no_grad():
        u_hat = solver.gibbs_initial_condition(
            theta=theta, n_burnin=burnin, n_samples=1, thin=thin
        )

    print(f"Running {N} steps ...")
    frames, energies = [], []
    H_init = solver.hamiltonian(u_hat).item()

    with torch.no_grad():
        u_hat_cur = u_hat.clone()
        for step in range(N + 1):
            frames.append(solver.spec2phys(u_hat_cur).numpy())
            energies.append(solver.energy(u_hat_cur).item())
            if step < N:
                u_hat_cur = solver.timestep(u_hat_cur, nsteps=nsteps)

    H_final = solver.hamiltonian(u_hat_cur).item()
    U = np.stack(frames, axis=0)   # (N+1, J)
    t_arr = np.arange(N + 1) * dt_snapshot
    x_arr = solver.x.numpy()

    print(f"Mean energy : {np.mean(energies):.6f}  (should be ~1.0)")
    print(f"H drift     : {abs(H_final - H_init):.2e}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 5),
                             gridspec_kw={"height_ratios": [3, 1.5]})

    vmax = np.abs(U).max()
    pcm = axes[0].imshow(
        U.T, aspect="auto", origin="lower",
        extent=[t_arr[0], t_arr[-1], x_arr[0], x_arr[-1]],
        cmap="seismic", vmin=-vmax, vmax=vmax, interpolation="bicubic",
    )
    axes[0].set_ylabel("x")
    axes[0].set_title(f"{label}  [D={D}, E0={E0}, L0={L0}, θ={theta}]")
    fig.colorbar(pcm, ax=axes[0])

    ix0 = int(np.argmin(np.abs(x_arr)))
    axes[1].plot(t_arr, U[:, ix0], lw=0.8, color="steelblue")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("u")
    axes[1].set_title("Time series of u at x=0")
    axes[1].set_xlim(t_arr[0], t_arr[-1])

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved to {out_png}")


for args in CASES:
    run_case(*args)
