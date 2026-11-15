import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kdv.tkdv_solver import TKdVSolver

J      = 32
E0     = 100.0
L0     = 6.0
D      = 0.24
theta  = -0.5
burnin = 10000
thin   = 10

dt_solver           = 0.001
dt_snapshot         = 2.0
n_steps             = 100
sub_steps           = int(dt_snapshot / dt_solver)

out_png = os.path.join(os.path.dirname(__file__), "long_trajectory_downstream_th05.png")

device = "cuda" if torch.cuda.is_available() else "cpu"
solver = TKdVSolver(J=J, dt=dt_solver, D=D, E0=E0, L0=L0).to(device)

print("Sampling Gibbs IC ...")
with torch.no_grad():
    u_hat = solver.gibbs_initial_condition(
        theta=theta, n_burnin=burnin, n_samples=1, thin=thin
    )

print(f"Running {n_steps} snapshots x {sub_steps} sub-steps = T={n_steps * dt_snapshot} ...")
frames, energies = [], []
H_init = solver.hamiltonian(u_hat).item()

with torch.no_grad():
    u_hat_cur = u_hat.clone()
    for step in range(n_steps + 1):
        frames.append(solver.spec2phys(u_hat_cur).cpu().numpy())
        energies.append(solver.energy(u_hat_cur).item())
        if step < n_steps:
            print(f"  snapshot {step+1}/{n_steps} ...")
            u_hat_cur = solver.timestep(u_hat_cur, nsteps=sub_steps)

H_final = solver.hamiltonian(u_hat_cur).item()
U = np.stack(frames, axis=0)
t_arr = np.arange(n_steps + 1) * dt_snapshot
x_arr = solver.x.cpu().numpy()

print(f"Mean energy : {np.mean(energies):.6f}  (should be ~1.0)")
print(f"H drift     : {abs(H_final - H_init):.2e}")

fig, axes = plt.subplots(3, 1, figsize=(14, 8),
                         gridspec_kw={"height_ratios": [3, 1.5, 1.5]})

vmax = np.abs(U).max()
pcm = axes[0].contourf(
    t_arr, x_arr, U.T,
    levels=50, cmap="RdYlBu_r", vmin=-vmax, vmax=vmax,
)
axes[0].set_ylabel("x")
axes[0].set_title(f"Long trajectory  [D={D}, E0={E0}, L0={L0}, θ={theta}, T={t_arr[-1]:.0f}]")
fig.colorbar(pcm, ax=axes[0])

ix0 = int(np.argmin(np.abs(x_arr)))
axes[1].plot(t_arr, U[:, ix0], lw=0.8, color="steelblue")
axes[1].set_xlabel("time")
axes[1].set_ylabel("u")
axes[1].set_title("Time series of u at x=0")
axes[1].set_xlim(t_arr[0], t_arr[-1])

mid = n_steps // 2
axes[2].plot(x_arr, U[mid, :], lw=1.2, color="crimson")
axes[2].set_xlabel("x")
axes[2].set_ylabel("u")
axes[2].set_title(f"Domain slice u(x) at t={t_arr[mid]:.0f} (T/2)")
axes[2].set_xlim(x_arr[0], x_arr[-1])
axes[2].axhline(0, color="k", lw=0.5, ls="--")

plt.tight_layout()
plt.savefig(out_png, dpi=150)
print(f"Saved to {out_png}")
