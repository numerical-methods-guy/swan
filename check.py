# quick_check_solver_constants.py
from torch_harmonics.examples import PdeDataset
import torch
from definitions import earth_radius, rotation_speed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = PdeDataset(dt=3600, nsteps=1, dims=(128,256), grid="equiangular", normalize=True, device=device)
solver = ds.solver

print("definitions.py earth_radius:", earth_radius)
print("definitions.py rotation_speed:", rotation_speed)

print("solver attrs that look relevant:")
for name in ["radius","a","earth_radius","Omega","omega","rotation_speed","f0"]:
    if hasattr(solver, name):
        print(name, "=", getattr(solver, name))
