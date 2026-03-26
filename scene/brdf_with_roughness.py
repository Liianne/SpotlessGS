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
        self.output_dim = 3  # RGB reflectance

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


class DisneySpecular(BRDFModel):
    def __init__(self, roughness: float = 0.5):
        super().__init__()
        self.roughness = nn.Parameter(torch.tensor(roughness), requires_grad=True)

    def fresnel_schlick(self, cos_theta: Tensor, F0: Tensor):
        return F0 + (1.0 - F0) * torch.pow(1.0 - cos_theta, 5)

    def smith_geo(self, cos_theta: Tensor, roughness: Tensor) -> Tensor:
        a = roughness ** 2
        tan_theta = torch.sqrt(1.0 - cos_theta ** 2) / cos_theta
        return (2 * cos_theta) / (1 + torch.sqrt(1 + a * tan_theta ** 2))

    def microfacet(self, normal: Tensor, view_dir: Tensor, light_dir: Tensor) -> Tensor:
        half_vector = F.normalize(view_dir + light_dir, dim=-1)
        cos_h = torch.sum(normal * half_vector, dim=-1)
        cos_h = torch.clamp(torch.sum(normal * half_vector, dim=-1), min=0.001)
        return torch.exp(-1 / (cos_h ** 2 * self.roughness ** 2)) / (cos_h ** 2 + 1e-7)

    def forward(self, view_dir: Tensor, normal: Tensor, light_dir: Tensor) -> Tensor:
        normal = F.normalize(normal, dim=-1)
        light_dir = F.normalize(light_dir, dim=-1)
        view_dir = F.normalize(view_dir, dim=-1)

        # #Lambertian diffusion term
        # diffuse = self.diffuse(view_dir, normal, light_dir)

        #Specular term using Cook-Torrance microfacet model
        F0 = 0.04
        #cos_theta = torch.sum(view_dir * normal, dim=-1)
        cos_theta = torch.clamp(torch.sum(view_dir * normal, dim=-1), min=0.001)
        Fr = self.fresnel_schlick(cos_theta, F0)

        #Microfacet distribution term
        D = self.microfacet(normal, view_dir, light_dir)

        #Geometry term (Smith visibility function)
        G = self.smith_geo(cos_theta, self.roughness)

        # Specular term (Cook-Torrance)
        specular = (D * G * Fr) / (4 * cos_theta + 1e-7)

        return specular

class DisneyBRDF(BRDFModel):
    def __init__(self, albedo: Tensor, roughness: float = 0.5):
        super().__init__()
        self.diffuse_model = DisneyDiffuse()
        self.specular_model = DisneySpecular(roughness)
        self.albedo = albedo

    def forward(self, view_dir: Tensor, normal: Tensor, light_dir: Tensor) -> Tensor:
        diffuse = self.diffuse_model(view_dir, normal, light_dir) * self.albedo
        specular = self.specular_model(view_dir, normal, light_dir)

        return diffuse + specular
    
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
            return DisneyBRDF(albedo=albedo) 
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