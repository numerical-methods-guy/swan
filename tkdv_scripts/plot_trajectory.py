import sys
sys.path.insert(0, r"C:\Users\neeong\swan")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kdv.tkdv_solver import TKdVSolver

# --- Parameters (Qi & Majda 2019, upstream case) ---
J       = 32
dt      = 0.01
D       = 1.0
E0      = 100.0
L0      = 6.0
theta   = -0.1
N       = 200        # number of steps to save (T = N * dt = 2.0)
burnin  = 1000
thin    = 10

OUT_PNG = r"C:\Users\neeong\swan\tkdv_scripts\trajectory_D1.png"

# --- Build solver ---
solver = TKdVSolver(J=J, dt=dt, D=D, E0=E0, L0=L0)

# --- Generate Gibbs IC ---
print("Sampling Gibbs IC ...")
with torch.no_grad():
    u_hat = solver.gibbs_initial_condition(
        theta=theta, n_burnin=burnin, n_samples=1, thin=thin
    )

# --- Run trajectory ---
print(f"Running {N} steps ...")
frames = []
energies = []
H_init = solver.hamiltonian(u_hat).item()

with torch.no_grad():
    u_hat_cur = u_hat.clone()
    for step in range(N + 1):
        u = solver.spec2phys(u_hat_cur).numpy()   # (J,)
        frames.append(u)
        energies.append(solver.energy(u_hat_cur).item())
        if step < N:
            u_hat_cur = solver.timestep(u_hat_cur, nsteps=1)

H_final = solver.hamiltonian(u_hat_cur).item()

U = np.stack(frames, axis=0)   # (N+1, J)
t_arr = np.arange(N + 1) * dt
x_arr = solver.x.numpy()       # (J,)

print(f"Mean energy over trajectory : {np.mean(energies):.6f}  (should be ~1.0)")
print(f"Initial Hamiltonian         : {H_init:.6f}")
print(f"Final   Hamiltonian         : {H_final:.6f}")
print(f"Hamiltonian drift           : {abs(H_final - H_init):.2e}")

# --- Plot (matching Qi & Majda Fig 1 style) ---
fig, axes = plt.subplots(2, 1, figsize=(12, 5),
                         gridspec_kw={"height_ratios": [3, 1.5]})

# Top: space-time heatmap, x on y-axis, t on x-axis
vmax = np.abs(U).max()
pcm = axes[0].pcolormesh(
    t_arr, x_arr, U.T,
    cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto"
)
axes[0].set_ylabel("x")
axes[0].set_title(f"Time series of u(x,t)  [D={D}, E0={E0}, L0={L0}, θ={theta}]")
fig.colorbar(pcm, ax=axes[0])

# Bottom: time series at x=0 (index closest to 0)
ix0 = int(np.argmin(np.abs(x_arr)))
axes[1].plot(t_arr, U[:, ix0], lw=0.8, color="steelblue")
axes[1].set_xlabel("time")
axes[1].set_ylabel("u")
axes[1].set_title("Time series of u evaluated at x=0")
axes[1].set_xlim(t_arr[0], t_arr[-1])

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"Saved to {OUT_PNG}")
