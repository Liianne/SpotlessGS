from .brdf import BRDFFactory
from .light import LightFactory
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from lietorch import SO3
from scene.cameras import Camera
#from scene.shading import ShadingModel
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
import math
from utils.sh_utils import eval_sh

class ShadingModel(nn.Module):
    #def __init__(self, brdf: str = "Lambertian", light: str = "Gaussian1D", albedo: float = 100.,  device: str = "gpu") -> None: 
    #def __init__(self, brdf: str = "Lambertian", light: str = "1DMLP", albedo: float = 100.,  device: str = "gpu", relit: bool = False) -> None:
    def __init__(self, brdf: str = "MLP", light: str = "1DMLP", albedo: float = 100.,  device: str = "gpu", relit: bool = False) -> None:   
    #def __init__(self, brdf: str = "Disney", light: str = "1DMLP", albedo: float = 100.,  device: str = "gpu", relit: bool = False) -> None:
    #def __init__(self, brdf: str = "LambertianMLP", light: str = "1DMLP", albedo: float = 100.,  device: str = "gpu", relit: bool = False) -> None:      
        super(ShadingModel, self).__init__()
        self.light = LightFactory.get_light(light)
        self.brdf = BRDFFactory.get_brdf(brdf)
        self.albedo_log = nn.Parameter(torch.tensor(albedo), requires_grad=True)
        #self.brdf = BRDFFactory.get_brdf(brdf, albedo=torch.exp(self.albedo_log))
        self.ambient_light_log = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        # target's own coordinate system used as world coordinate. Right-Down-Forward. Hardcoded here that normal pointing from the origin of the target to camera.

        self.scaling_factor = nn.Parameter(torch.tensor(0.1), requires_grad=True) # When calibrating, should not optimize this scaling factor
        self._warmup_factor : float = 1.0 # DEPRECATED 

        #newly added: use spherical harmonics to represent ambient light
        #self.sh_coeffs = nn.Parameter(torch.zeros(25, 3), requires_grad=True) 
        self.sh_coeffs = nn.Parameter(torch.zeros(16, 3), requires_grad=True)

        self.relit = relit
        
        self.intensity_scale = nn.Parameter(torch.tensor(0.1), requires_grad=True)

        self.set_optimizer()

    def set_optimizer(self)->None:
        l = [
            #{'params': [self.ambient_light_log], 'lr': 0.001, "name": "ambient_light"},
            {'params': [self.scaling_factor], 'lr': 0.001, "name": "scaling"},
            # uncomment for extensive finetuning (experimental)
            {'params': [self.light.tau_log], 'lr': 0.001, "name": "tau"}, 
            {'params': [self.light.gamma_log], 'lr': 0.001, "name": "gamma"},
            #{'params': [self.light._r_l2c_SO3], 'lr': 0.001, "name": "r_vec"},
            {'params': [self.light._t_vec], 'lr': 0.001, "name": "_t_vec"},
            #{'params': [self.light.sigma], 'lr': 0.001, "name": "sigma"},
            #{'params': [self.light.mlp.parameters()], 'lr': 0.001, "name": "mlp0"},
            {'params': list(self.light.mlp.parameters()), 'lr': 0.001, "name": "mlp0"},
            {'params': list(self.brdf.parameters()), 'lr': 0.0001, "name": "brdf_mlp"},
            #{'params': self.sh_coeffs, 'lr': 0.001, "name": "sh_coeffs"},
            #{'params': list(self.light.mlp.parameters()), 'lr': 0.0001, "name": "mlp0"},
            #{'params': list(self.brdf.parameters()), 'lr': 0.0001, "name": "brdf_mlp"},
            #{'params': self.sh_coeffs, 'lr': 0.0001, "name": "sh_coeffs"},
            {'params': self.sh_coeffs, 'lr': 0.001, "name": "sh_coeffs"},
            #{'params': [self.intensity_scale], 'lr': 0.0001, "name": "intensity_scale"},
        ]

        # l = [
        #     #{'params': [self.ambient_light_log], 'lr': 0.001, "name": "ambient_light"},
        #     #{'params': [self.scaling_factor], 'lr': 0.001, "name": "scaling"},
        #     # uncomment for extensive finetuning (experimental)
        #     #{'params': [self.light.tau_log], 'lr': 0.001, "name": "tau"}, 
        #     #{'params': [self.light.gamma_log], 'lr': 0.001, "name": "gamma"},
        #     #{'params': [self.light._r_l2c_SO3], 'lr': 0.001, "name": "r_vec"},
        #     #{'params': [self.light._t_vec], 'lr': 0.001, "name": "_t_vec"},
        #     #{'params': [self.light.sigma], 'lr': 0.001, "name": "sigma"},
        #     #{'params': [self.light.mlp.parameters()], 'lr': 0.001, "name": "mlp0"},
        #     #{'params': list(self.light.mlp.parameters()), 'lr': 0.001, "name": "mlp0"},
        #     #{'params': list(self.brdf.parameters()), 'lr': 0.0001, "name": "brdf_mlp"},
        #     #{'params': self.sh_coeffs, 'lr': 0.001, "name": "sh_coeffs"},
        #     #{'params': list(self.light.mlp.parameters()), 'lr': 0.0001, "name": "mlp0"},
        #     #{'params': list(self.brdf.parameters()), 'lr': 0.0001, "name": "brdf_mlp"},
        #     #{'params': self.sh_coeffs, 'lr': 0.0001, "name": "sh_coeffs"},
        #     {'params': self.sh_coeffs, 'lr': 0.001, "name": "sh_coeffs"},
        #     #{'params': [self.intensity_scale], 'lr': 0.0001, "name": "intensity_scale"},
        # ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)


    @property
    def ambient_light(self):
        return torch.exp(self.ambient_light_log)
    

    @property
    def albedo(self):
        return torch.exp(self.albedo_log)
    
    def set_albedo(self, albedo: float, require_grad: bool = True) -> None:
        self.albedo_log = nn.Parameter(torch.log(torch.tensor(albedo)), requires_grad=require_grad)


    # DEPRECATED FUNCTION
    # @property
    # def warmup_factor(self)->float:
    #     return self._warmup_factor
    
    # # DEPRECATED FUNCTION
    # @warmup_factor.setter
    # def warmup_factor(self, value: float)->None:
    #     if 0. <= value <= 1.:
    #         self._warmup_factor = value
    #     else:
    #         raise ValueError(f"Invalid input for warmup factor {value}. Please check input.")

    def set_ambient_light(self, ambient_light: float, require_grad: bool = True) -> None:
        self.ambient_light_log = nn.Parameter(torch.log(torch.tensor(ambient_light)), requires_grad=require_grad)
        self.set_optimizer()
        
    def set_scaling_factor(self, scaling_factor: float, require_grad: bool = True) -> None:
        self.scaling_factor = nn.Parameter((torch.tensor(scaling_factor)), requires_grad=require_grad)
        self.set_optimizer()


    def factorial(self, n):
     
        """Compute factorial using PyTorch, ensuring proper handling of zero case."""
        n = torch.as_tensor(n, dtype=torch.float32, device="cuda")
        if n == 0:
            return torch.tensor(1.0, device=n.device, dtype=n.dtype)
        return torch.prod(torch.arange(1, n + 1, device=n.device, dtype=n.dtype))

    # def factorial(self, n):
    #     """Compute factorial using PyTorch (to keep everything on GPU)."""
    #     if n == 0 or n == 1:
    #         return torch.tensor(1.0, device=n.device)
    #     return torch.prod(torch.arange(1, n + 1, dtype=torch.float32, device=n.device))

    def legendre_polynomial(self, l, m, x):
        """Compute the associated Legendre polynomial P_l^m(x) using recursion."""
        if m < 0 or m > l:
            return torch.zeros_like(x)  # Undefined case
        if l == 0:
            return torch.ones_like(x)
        if l == 1:
            return x if m == 0 else -torch.sqrt(1 - x**2)

        p_mm = torch.ones_like(x)  # P_m^m(x)
        if m > 0:
            p_mm = (-1)**m * self.factorial(2 * m - 1) * (1 - x**2).pow(m / 2)

        if l == m:
            return p_mm

        p_mmp1 = x * (2 * m + 1) * p_mm  # P_(m+1)^m(x)

        if l == m + 1:
            return p_mmp1

        for n in range(m + 2, l + 1):
            p_n = ((2 * n - 1) * x * p_mmp1 - (n + m - 1) * p_mm) / (n - m)
            p_mm, p_mmp1 = p_mmp1, p_n

        return p_mmp1

    def real_spherical_harmonic(self, l, m, theta, phi):
        device = theta.device
        dtype = theta.dtype

        norm_factor = torch.sqrt((2 * l + 1) / (4 * math.pi) * self.factorial(l - abs(m)) / self.factorial(l + abs(m)))
        legendre = self.legendre_polynomial(l, abs(m), torch.cos(theta))

        sqrt2 = torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))

        if m > 0:
            return sqrt2 * norm_factor * legendre * torch.cos(m * phi)
        elif m < 0:
            return sqrt2 * norm_factor * legendre * torch.sin(-m * phi)
        else:
            return norm_factor * legendre


    def eval_sh_basis_poly(self, normals: Tensor, light_dir: Tensor) -> Tensor:
   
        x, y, z = normals[..., 0], normals[..., 1], normals[..., 2]
        vx, vy, vz = light_dir[..., 0], light_dir[..., 1], light_dir[..., 2]
    
        cos_theta = x * vx + y * vy + z * vz  
    
        sh_basis = torch.stack([
            torch.ones_like(x) * 0.2821,      # Y_0^0
            0.4886 * y,                       # Y_1^{-1}
            0.4886 * z,                       # Y_1^0
            0.4886 * x,                       # Y_1^1
            1.0925 * x * y,                   # Y_2^{-2}
            1.0925 * y * z,                   # Y_2^{-1}
            0.3154 * (3 * z**2 - 1),          # Y_2^0
            1.0925 * x * z,                   # Y_2^1
            0.5462 * (x**2 - y**2),           # Y_2^2
            0.5900 * y * (3 * x**2 - y**2),   # Y_3^{-3}
            2.8906 * x * y * z,               # Y_3^{-2}
            0.4570 * y * (5 * z**2 - 1),      # Y_3^{-1}
            0.3732 * (5 * z**3 - 3 * z),      # Y_3^0
            0.4570 * x * (5 * z**2 - 1),      # Y_3^1
            1.4453 * z * (x**2 - y**2),       # Y_3^2
            0.5900 * x * (x**2 - 3 * y**2)    # Y_3^3
        ], dim=-1) 
    
        sh_basis = sh_basis * cos_theta.unsqueeze(-1) 
    
        return sh_basis


    def eval_sh_basis_normals(self, normals: Tensor) -> Tensor:
        """
        Evaluate real SH basis (l <= 3 → 16 coeffs) on surface normals.
        normals: [N, 3]
        return:  [N, 16]
        """
        x, y, z = normals[..., 0], normals[..., 1], normals[..., 2]

        sh = torch.stack([
            # l = 0
            0.282095 * torch.ones_like(x),

            # l = 1
            0.488603 * y,
            0.488603 * z,
            0.488603 * x,

            # l = 2
            1.092548 * x * y,
            1.092548 * y * z,
            0.315392 * (3*z*z - 1),
            1.092548 * x * z,
            0.546274 * (x*x - y*y),

            # l = 3
            0.590044 * y * (3*x*x - y*y),
            2.890611 * x * y * z,
            0.457046 * y * (5*z*z - 1),
            0.373176 * (5*z*z*z - 3*z),
            0.457046 * x * (5*z*z - 1),
            1.445306 * z * (x*x - y*y),
            0.590044 * x * (x*x - 3*y*y),
        ], dim=-1)

        return sh



    # def spherical_harmonic(self, l, m, theta, phi):
    #     """Evaluate spherical harmonic Y^l_m for given angles theta and phi."""
    #     # Detach tensors from the computation graph and move to CPU before converting to numpy
    #     theta_cpu = theta.detach().cpu().numpy() if theta.is_cuda else theta.detach().numpy()
    #     phi_cpu = phi.detach().cpu().numpy() if phi.is_cuda else phi.detach().numpy()

    #     return sph_harm(m, l, phi_cpu, theta_cpu)

    def spherical_harmonic(self, l, m, theta, phi):
        """Evaluate spherical harmonic Y^l_m for given angles theta and phi using PyTorch."""
        
        # Ensure m and l are integers
        l = torch.tensor(l, dtype=torch.float32)  # Ensure l is a tensor
        m = torch.tensor(m, dtype=torch.float32)  # Ensure m is a tensor

        # Precompute the normalization factor
        norm_factor = torch.sqrt((2 * l + 1) / (4 *torch.pi) * self.factorial(l - m) / self.factorial(l + m))
        
        # Compute associated Legendre polynomial P_l^m(cos(theta))
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        # Compute the associated Legendre polynomial recursively
        P_lm = torch.ones_like(cos_theta)  # Start with P_l^m as 1 (the lowest order)

        l = int(l)
        m = int(m)
        
        if m > 0:
            # Recursively compute the associated Legendre polynomials P_l^m(cos(theta))
            for i in range(m):
                P_lm *= (2 * i + 1) * torch.sqrt(1 - cos_theta**2)
        
        # Compute the spherical harmonic function
        Y_lm = norm_factor * P_lm * torch.exp(1j * m * phi)
        
        return Y_lm



    def eval_sh_basis(self, normals: Tensor, light_dir: Tensor) -> Tensor:
        """Evaluate spherical harmonic basis functions for normals and light directions."""
        # Convert normals and light directions to spherical coordinates (theta, phi)
        theta_normal, phi_normal = torch.acos(normals[..., 2]), torch.atan2(normals[..., 1], normals[..., 0])
        theta_light, phi_light = torch.acos(light_dir[..., 2]), torch.atan2(light_dir[..., 1], light_dir[..., 0])
        
        x, y, z = normals[..., 0], normals[..., 1], normals[..., 2]

        # Evaluate SH basis for normal direction
        sh_values_normal = []
        sh_values_light = []
    
        for l in range(0, 5):  # Using l = 0 to 3 for SH basis functions
            for m in range(-l, l + 1):
                # sh_value = self.spherical_harmonic(l, m, theta_normal, phi_normal)
                # #sh_values_normal.append(sh_value)
                #  # Convert the sh_value (numpy.ndarray) to a torch.Tensor
                # sh_values_normal.append(torch.tensor(sh_value, dtype=torch.float32))

                # Compute the spherical harmonics for the normal directions
                sh_normal = self.real_spherical_harmonic(l, m, torch.acos(z), torch.atan2(y, x))
                sh_values_normal.append(sh_normal)
                
                # Compute the spherical harmonics for the light directions
                sh_light = self.real_spherical_harmonic(l, m, torch.acos(light_dir[..., 2]), torch.atan2(light_dir[..., 1], light_dir[..., 0]))
                sh_values_light.append(sh_light)

        # sh_values_normal = torch.stack(sh_values_normal, dim=-1)

        # # Evaluate SH basis for light direction
        # sh_values_light = []
        # for l in range(0, 4):  # Using l = 0 to 3 for SH basis functions
        #     for m in range(-l, l + 1):
        #         sh_value = self.spherical_harmonic(l, m, theta_light, phi_light)
        #         #sh_values_light.append(sh_value)
        #         sh_values_light.append(torch.tensor(sh_value, dtype=torch.float32))

        # sh_values_light = torch.stack(sh_values_light, dim=-1)


        # Stack spherical harmonic values for normal and light directions
        sh_values_normal = torch.stack(sh_values_normal, dim=-1)  # Shape: (N, SH_terms)
        sh_values_light = torch.stack(sh_values_light, dim=-1)  # Shape: (N, SH_terms)


        return sh_values_normal, sh_values_light

    def get_SH(self):
        return self.sh_coeffs.detach().cpu().clone()

    def set_SH(self, new_sh: torch.Tensor):
        with torch.no_grad():
            self.sh_coeffs.copy_(new_sh.to(self.sh_coeffs.device))  


    def forward(self, pts: Tensor, camera: Camera, albedos: Tensor, normals: Tensor, roughness: Tensor, metalness: Tensor)-> Tensor:
        '''
        Arguments:
            pts: 3D points in world coordinate.
        '''
        # Monocular SfM poses are up-to-scale. So need to optimize for scale here.
        pts = torch.nan_to_num(pts, nan=0.0)

        pts_scaled = pts*self.scaling_factor

        #print("Scale factor: ", self.scaling_factor)

        
        t_w2c: Tensor = camera.world_view_transform[3:4, :3]*self.scaling_factor
        rmat_w2c: Tensor = camera.world_view_transform[:3, :3]
        t_c2w = -torch.matmul(rmat_w2c, t_w2c.transpose(0,1))
        t_l2c = self.light.t_l2c().squeeze(0).transpose(0,1) 
        p_l_in_w = torch.matmul(rmat_w2c, t_l2c)+t_c2w
        light_dir = -p_l_in_w.transpose(0, 1)+pts_scaled
        view_dir = t_c2w.transpose(0,1)-pts_scaled

        pts_in_cam = torch.matmul(pts_scaled, rmat_w2c)+t_w2c

        

        #pts_in_cam = pts_in_cam / (pts_in_cam[..., 2:3] + 1e-6)
        #print("The size of x is: ", x_in_l.size())

        #pts_in_cam_x_y = (torch.square(pts_in_cam[..., 0])+torch.square(pts_in_cam[..., 1])).to(torch.float32)[..., None]


    #     #newly added
    #     pts_in_cam_x_y = torch.clamp(pts_in_cam_x_y, min=0)

    #     pts_in_cam_x_y = torch.sqrt(pts_in_cam_x_y)

    #    # Sort by distance for cleaner plot
    #     sorted_idx = pts_in_cam_x_y.argsort(dim=0, descending=False).squeeze()

    #     # Squeeze to 1D if needed
    #     x_vals = pts_in_cam_x_y[sorted_idx].squeeze().detach().cpu().numpy()
    #     y_vals = incident_light[sorted_idx].squeeze().detach().cpu().numpy()

    #     plt.figure(figsize=(6, 4))
    #     plt.plot(x_vals, y_vals)
    #     plt.xlabel("Distance from Camera Center")
    #     plt.ylabel("Incident Light Intensity")
    #     plt.title("Radial Light Falloff Curve")
    #     plt.grid(True)

    #     # Save the figure
    #     plt.savefig("/home/stud/hlia/storage/master_thesis/ori_ori_darkgs/darkgs/output/curves/radial_light_falloff.png", dpi=300, bbox_inches='tight')
    #     print("Saved rid curve!")
    #     plt.close()  # Close the figure to avoid memory leaks in long loops
                
        normals = F.normalize(normals, dim=-1, eps=1e-6)

        light_dir = F.normalize(light_dir, dim=-1, eps=1e-6)

        sh_value = self.eval_sh_basis_poly(normals, light_dir)

        ambient_light = torch.matmul(sh_value, self.sh_coeffs) #* torch.matmul(sh_values_light, self.sh_coeffs)
        # sh_value = self.eval_sh_basis_normals(normals)    # [N,16]
        # ambient_light = sh_value @ self.sh_coeffs         # [N,3]


        #use this for mlp_based brdf
        # reflectance = self.brdf(view_dir, light_dir, normals, metalness, roughness)[..., None] # n*1
        # reflectance = reflectance.squeeze(-1)

        reflectance = self.brdf(view_dir, light_dir, normals, albedos, metalness, roughness)[..., None]
        reflectance = reflectance.squeeze(-1)



        # reflectance = self.brdf(view_dir, normals, light_dir, metalness, roughness)[..., None] # n*1
        # reflectance = reflectance.squeeze(-1)
        #print("The shape of reflectance: ", reflectance.size())

        #reflectance = self.brdf(view_dir, light_dir, normals)[..., None] # n*1
        #reflectance = reflectance.squeeze(-1)

        # reflectance = self.brdf(view_dir, light_dir, normals)[..., None] # n*1
        # reflectance = reflectance.squeeze(-1)

        #reflectance = self.brdf(view_dir, normals, light_dir)[..., None] # n*1

        # reflectance = self.brdf(view_dir, normals, light_dir, F.softplus(albedos), roughness, metalness)[..., None] # n*1
        # reflectance = reflectance.squeeze(-1)
      
        
     
     
     
        incident_light = self.light(pts_in_cam.unsqueeze(0)).transpose(0,1)
    
    
        #print("Albedo: ", albedos[:5])
        #print("Normals: ", normals)
        #original:
        reflected_light = F.softplus(albedos) * (incident_light+ ambient_light) * reflectance #* self.intensity_scale
        #reflected_light = F.softplus(albedos) * (0.5 + ambient_light) * reflectance
        #reflected_light = F.softplus(albedos) * (incident_light+ self.ambient_light) * reflectance #* self.intensity_scale
        #reflected_light = F.softplus(albedos) * (incident_light) * reflectance
        #reflected_light = F.softplus(albedos) * ambient_light * reflectance #* self.intensity_scale
        
        #reflected_light = F.softplus(albedos) * (incident_light)
        
        
        #reflected_light = F.softplus(albedos) * 0.2
        #reflected_light = F.softplus(albedos) * 0.05
        #reflected_light = incident_light * reflectance 
        #reflected_light = F.softplus(albedos) * (ambient_light + 0.3) * reflectance  #* self.intensity_scale
        #reflected_light = F.softplus(albedos) * incident_light * reflectance
        #reflected_light = F.softplus(albedos) * (ambient_light + 0.01) * reflectance
        #reflected_light = F.softplus(albedos) * ambient_light * reflectance
        #reflected_light = F.softplus(albedos) * (ambient_light + 0.3) * reflectance
        #reflected_light = F.softplus(albedos) * reflectance

        
        
        return reflected_light
        #return torch.cat([roughness, torch.zeros_like(roughness), torch.zeros_like(roughness)], dim=-1)
    