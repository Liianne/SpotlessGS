import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math

class BRDFModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, view_dir: Tensor, normal: Tensor, light_dir: Tensor)->Tensor:
        view_dir = F.normalize(view_dir, dim=0)
        normal = F.normalize(normal, dim=0)
        light_dir = F.normalize(light_dir, dim=0)

class Lambertian(BRDFModel):
    def forward(self, view_dir: Tensor, normal: Tensor, light_dir: Tensor)->Tensor:
        # Lambertian Cos Law
        normal = F.normalize(normal, dim=-1)
        light_dir = F.normalize(light_dir, dim=-1)
        
        # print(f"light_dir shape: {light_dir.shape}")
        # print(f"normal shape: {normal.shape}")

        cos = torch.relu(torch.sum(light_dir*normal, dim = -1))
        return cos

#Lambertian and Blinn-Phong
class BlinnPhong(BRDFModel):
    def __init__(self, specular_color: float = 0.5, shininess: float = 32.0):
        super().__init__()
        # self.specular_color = specular_color # cs
        # self.shininess = shininess # m

        #set the two parameters to be learnable
        self.specular_color = nn.Parameter(torch.tensor(specular_color), requires_grad=True)  
        self.shininess = nn.Parameter(torch.tensor(shininess), requires_grad=True)  

    def forward(self, view_dir: Tensor, normal: Tensor, light_dir: Tensor, albedo: Tensor)->Tensor:

        normal = F.normalize(normal, dim=-1)
        light_dir = F.normalize(light_dir, dim=-1)
        view_dir = F.normalize(view_dir, dim=-1)

        #Lambertian diffuse term
        #cos = torch.relu(torch.sum(light_dir*normal, dim = -1))
        diffuse = albedo * torch.relu(torch.sum(light_dir * normal, dim = -1, keepdim=True))

        #BlinnPhong specular term
        halfway = F.normalize(light_dir + view_dir, dim=-1)
        specular = torch.pow(torch.relu(torch.sum(normal * halfway, dim=-1)), self.shininess)

        # Broadcast specular to (N, 3)
        specular = specular.unsqueeze(-1)
        specular = specular.expand(-1, 3) 

       

        # Output the size of the tensors
        print(f"Diffuse size: {diffuse.shape}")
        print(f"Specular size: {specular.shape}")


        

        return diffuse + specular
class PositionalEncoding:
    def __init__(self, num_frequencies=6):
        self.num_frequencies = num_frequencies

    def encode(self, x):
        encodings = [x]  
        for i in range(self.num_frequencies):
            encodings.append(torch.sin(2.0 ** i * math.pi * x))
            encodings.append(torch.cos(2.0 ** i * math.pi * x))
        return torch.cat(encodings, dim=-1)

# class MLP_BRDF(nn.Module):
#     def __init__(self, input_dim=9, hidden_dim=256, output_dim=3, num_frequencies=6):
#         super(MLP_BRDF, self).__init__()

#         self.pe = PositionalEncoding(num_frequencies)
#         self.pe_dim = input_dim * (2 * num_frequencies + 1) 

#         self.fc1 = nn.Linear(self.pe_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc4 = nn.Linear(hidden_dim, output_dim)

#         self.norm1 = nn.LayerNorm(hidden_dim)
#         self.norm2 = nn.LayerNorm(hidden_dim)
#         self.norm3 = nn.LayerNorm(hidden_dim)
        
#         self.skip = nn.Linear(self.pe_dim, hidden_dim) 

#     def forward(self, x):
#         x = self.pe.encode(x)  
#         x_skip = self.skip(x)  

#         x = F.softplus(self.fc1(x), beta=10)
#         x = self.norm1(x)
        
#         x = F.softplus(self.fc2(x) + x_skip, beta=10)  
#         x = self.norm2(x)
        
#         x = F.softplus(self.fc3(x), beta=10)
#         x = self.norm3(x)
        
#         x = torch.sigmoid(self.fc4(x)) * 2  

#         return x

class MLP_BRDF(nn.Module):
    def __init__(self, num_frequencies=6, hidden_dim=256):
        super(MLP_BRDF, self).__init__()
        self.encoder = PositionalEncoding(num_frequencies)
        
        self.dir_dim = 9  # view_dir + light_dir + normal (3x3)
        self.pe_dim = self.dir_dim * (2 * num_frequencies + 1)
        self.input_dim = self.pe_dim + 2  # +2 for roughness and metalness
        #self.output_dim = 3  # RGB reflectance
        self.output_dim = 1

        # MLP definition
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.output_dim),
            nn.Sigmoid()  # reflectance in [0,1]
        )

    def forward(self, view_dir, light_dir, normal, roughness, metalness):
        # Normalize directions
        view_dir = F.normalize(view_dir, dim=-1)
        light_dir = F.normalize(light_dir, dim=-1)
        normal = F.normalize(normal, dim=-1)

        # Concatenate and encode directional info
        x = torch.cat([view_dir, light_dir, normal], dim=-1)
        x = self.encoder.encode(x)

        # # Add roughness and metalness to each sample
        # batch_size = x.shape[0]
        # brdf_params = torch.cat([
        #     self.roughness.expand(batch_size, 1),
        #     self.metalness.expand(batch_size, 1)
        # ], dim=-1)

        roughness = roughness.to(x.device)
        metalness = metalness.to(x.device)


         # Concatenate BRDF parameters
        if roughness.dim() == 1:
            roughness = roughness.unsqueeze(-1)
        if metalness.dim() == 1:
            metalness = metalness.unsqueeze(-1)

        #print("roughness is: ", roughness)

        # Final input to MLP
        x = torch.cat([x, roughness, metalness], dim=-1)

        reflectance = self.net(x)  # Output: RGB reflectance

        #cos = torch.relu(torch.sum(light_dir*normal, dim = -1))

        #cos = torch.clamp((light_dir * normal).sum(dim=-1, keepdim=True), min=0.0)

        #return reflectance
        #reflectance = 0.8 * cos + 0.2 * reflectance
        return reflectance

    # def get_properties(self):
    #     return {
    #         "roughness": self.roughness.item(),
    #         "metalness": self.metalness.item()
    #     }

class DisneyDiffuse(BRDFModel):
    def forward(self, view_dir: Tensor, normal: Tensor, light_dir: Tensor)->Tensor:
        normal = F.normalize(normal, dim=-1)
        light_dir = F.normalize(light_dir, dim=-1)

        #Lambertian diffusion term
        cos = torch.relu(torch.sum(light_dir * normal, dim=-1))
        return cos


class DisneyBRDF(BRDFModel):
    def __init__(self):
        super().__init__()

    def fresnel_schlick(self, cos_theta: Tensor, F0: Tensor) -> Tensor:
        return F0 + (1.0 - F0) * (1.0 - cos_theta).clamp(0, 1).pow(5)

    def distribution_ggx(self, N: Tensor, H: Tensor, roughness: Tensor) -> Tensor:
        a = roughness ** 2
        NdotH = (N * H).sum(dim=-1, keepdim=True).clamp(1e-4, 1.0)
        denom = (NdotH ** 2 * (a ** 2 - 1.0) + 1.0)
        D = a ** 2 / (math.pi * denom ** 2 + 1e-5)
        return D

    def geometry_smith(self, N: Tensor, V: Tensor, L: Tensor, roughness: Tensor) -> Tensor:
        def G1(N, X):
            NdotX = (N * X).sum(dim=-1, keepdim=True).clamp(1e-4, 1.0)
            r = roughness + 1.0
            k = (r ** 2) / 8.0
            return NdotX / (NdotX * (1.0 - k) + k + 1e-5)

        return G1(N, V) * G1(N, L)

    def forward(
        self,
        view_dir: Tensor,
        normal: Tensor,
        light_dir: Tensor,
        base_color: Tensor,
        roughness: Tensor,
        metalness: Tensor
    ) -> Tensor:
        # Normalize inputs
        N = F.normalize(normal, dim=-1)
        V = F.normalize(view_dir, dim=-1)
        L = F.normalize(light_dir, dim=-1)
        H = F.normalize(V + L, dim=-1)

        NdotL = (N * L).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        NdotV = (N * V).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)

        # Base reflectivity
        F0 = torch.lerp(torch.full_like(base_color, 0.04), base_color, metalness)

        # Fresnel
        F_term = self.fresnel_schlick((H * V).sum(dim=-1, keepdim=True).clamp(0.0, 1.0), F0)

        # Specular
        D = self.distribution_ggx(N, H, roughness)
        G = self.geometry_smith(N, V, L, roughness)
        specular = (D * G * F_term) / (4.0 * NdotL * NdotV + 1e-5)

        # Diffuse
        kd = (1.0 - F_term) * (1.0 - metalness)
        diffuse = kd * base_color / math.pi

        return (diffuse + specular) * NdotL

class LambertianMLP(nn.Module):
    def __init__(self, mlp_hidden_dim=128, mlp_freqs=6, residual_weight=0.3):
        super().__init__()
        self.lambertian = Lambertian()
        self.residual_weight = residual_weight

        # --- Residual MLP branch ---
        self.encoder = PositionalEncoding(num_frequencies=mlp_freqs)
        self.dir_dim = 9  # view_dir + light_dir + normal (3x3)
        self.pe_dim = self.dir_dim * (2 * mlp_freqs + 1)
        self.input_dim = self.pe_dim + 2  # +2 for roughness + metalness
        self.output_dim = 1  # scalar shading term

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, self.output_dim),
            nn.Tanh()  # residual shading in [-1,1]
        )

    def forward(
        self,
        view_dir: Tensor,
        normal: Tensor,
        light_dir: Tensor,
        roughness: Tensor,
        metalness: Tensor
    ) -> Tensor:
        # --- Lambertian base shading ---
        lambertian_term = self.lambertian(view_dir, normal, light_dir)  # [N] scalar

        # --- Residual MLP shading ---
        view_dir = F.normalize(view_dir, dim=-1)
        light_dir = F.normalize(light_dir, dim=-1)
        normal = F.normalize(normal, dim=-1)

        # Encode angular info
        x = torch.cat([view_dir, light_dir, normal], dim=-1)
        x = self.encoder.encode(x)

        # Add roughness + metalness
        if roughness.dim() == 1:
            roughness = roughness.unsqueeze(-1)
        if metalness.dim() == 1:
            metalness = metalness.unsqueeze(-1)
        x = torch.cat([x, roughness, metalness], dim=-1)

        residual = self.mlp(x).squeeze(-1) * self.residual_weight

        # Final scalar shading term
        shading = torch.clamp(lambertian_term + residual, 0.0, 1.0)
        return shading

    
class BRDFFactory:
    @staticmethod
    #def get_brdf(BRDF_type, albedo) -> BRDFModel:
    def get_brdf(BRDF_type) -> BRDFModel:
        if BRDF_type == "Lambertian":
            return Lambertian()
        elif BRDF_type == "BlinnPhong":
            return BlinnPhong()
        elif BRDF_type == "MLP":
            return MLP_BRDF()
        elif BRDF_type == "Disney":
            return DisneyBRDF() 
        elif BRDF_type == "LambertianMLP":
            return LambertianMLP()
        else:
            raise ValueError(f"BRDF type {BRDF_type} not recognized!")


# unit test code
if __name__ == "__main__":
    brdf = BRDFFactory.get_brdf("DisneyDiffuse")

    x = torch.linspace(-1, 1, 50)

    for aa in torch.linspace(-1.0, 1.0, 19):
        s = []
        brdf._roughness.data = aa
        print(aa)
        for xx in x:
            theta = xx*torch.pi/2
            print("SIN: ", torch.sin(theta))
            light_dir = torch.tensor([[torch.abs(torch.sin(theta))],[0],[-torch.cos(theta)]], dtype=float)
            normal = torch.tensor([0,0,-1.0], dtype=float)
            view_dir = torch.tensor([[0],[0],[-.68]], dtype=float)
            ss = brdf(view_dir, normal, light_dir)
            s.append(ss[0])
        
        s = torch.tensor(s)
        import matplotlib.pyplot as plt

        plt.plot(x, s)
    plt.show()