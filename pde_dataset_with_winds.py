import torch
from torch_harmonics.examples import PdeDataset


class PdeDatasetWithWinds(torch.utils.data.Dataset):
    """
    Extended dataset that computes physical winds for paradis architecture.
    
    This dataset wraps PdeDataset and computes physical winds (u, v) from the
    spectral vorticity and divergence using the getuv method. The winds are:
    1. Computed from denormalized spectral coefficients
    2. Normalized separately using wind-specific statistics
    """
    
    def __init__(
        self,
        dt,
        nsteps,
        dims=(384, 768),
        grid="equiangular",
        pde="shallow water equations",
        initial_condition="random",
        num_examples=32,
        device=torch.device("cpu"),
        normalize=True,
        stream=None,
    ):
        self.base_dataset = PdeDataset(
            dt=dt,
            nsteps=nsteps,
            dims=dims,
            grid=grid,
            pde=pde,
            initial_condition=initial_condition,
            num_examples=num_examples,
            device=device,
            normalize=normalize,
            stream=stream,
        )
        
        self.solver = self.base_dataset.solver
        self.nlat = self.base_dataset.nlat
        self.nlon = self.base_dataset.nlon
        # FIX: Use the 'grid' argument directly instead of accessing it from base_dataset
        self.grid = grid
        self.nsteps = self.base_dataset.nsteps
        self.normalize = normalize
        self.device = device
        self.ictype = self.base_dataset.ictype
        
        self.inp_mean = self.base_dataset.inp_mean
        self.inp_var = self.base_dataset.inp_var
        
        if self.normalize:
            self._compute_wind_statistics()
    
    def _compute_wind_statistics(self, num_samples=100):
        """Compute mean and variance for wind normalization."""
        wind_samples = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                if self.ictype == "random":
                    inp_spec = self.solver.random_initial_condition(mach=0.2)
                elif self.ictype == "galewsky":
                    inp_spec = self.solver.galewsky_initial_condition()
                
                winds = self.solver.getuv(inp_spec[1:])
                wind_samples.append(winds)
        
        wind_samples = torch.stack(wind_samples, dim=0)
        
        self.wind_mean = torch.mean(wind_samples, dim=(0, 2, 3), keepdim=True).reshape(2, 1, 1)
        self.wind_var = torch.var(wind_samples, dim=(0, 2, 3), keepdim=True).reshape(2, 1, 1)
        
        self.wind_var = torch.maximum(self.wind_var, torch.ones_like(self.wind_var) * 1e-8)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def set_initial_condition(self, ictype="random"):
        """Set the initial condition type."""
        self.base_dataset.set_initial_condition(ictype)
        self.ictype = ictype
    
    def set_num_examples(self, num_examples=32):
        """Set the number of examples."""
        self.base_dataset.set_num_examples(num_examples)
    
    def _get_sample_with_winds(self):
        """Generate a sample with both fields and winds."""
        if self.ictype == "random":
            inp_spec = self.solver.random_initial_condition(mach=0.2)
        elif self.ictype == "galewsky":
            inp_spec = self.solver.galewsky_initial_condition()
        
        tar_spec = self.solver.timestep(inp_spec, self.nsteps)
        
        inp_fields = self.solver.spec2grid(inp_spec)
        tar_fields = self.solver.spec2grid(tar_spec)
        
        inp_winds = self.solver.getuv(inp_spec[1:])
        tar_winds = self.solver.getuv(tar_spec[1:])
        
        return inp_fields, inp_winds, tar_fields, tar_winds
    
    def __getitem__(self, index):
        """Get a sample with fields and winds."""
        with torch.inference_mode():
            with torch.no_grad():
                inp_fields, inp_winds, tar_fields, tar_winds = self._get_sample_with_winds()
                
                if self.normalize:
                    inp_fields = (inp_fields - self.inp_mean) / torch.sqrt(self.inp_var)
                    tar_fields = (tar_fields - self.inp_mean) / torch.sqrt(self.inp_var)
                    
                    inp_winds = (inp_winds - self.wind_mean) / torch.sqrt(self.wind_var)
                    tar_winds = (tar_winds - self.wind_mean) / torch.sqrt(self.wind_var)
        
        return inp_fields.clone(), inp_winds.clone(), tar_fields.clone(), tar_winds.clone()
