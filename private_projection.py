import numpy as np
import torch
from scipy.stats import norm
from scipy.optimize import minimize
from scipy import linalg
from randomized_response import randomize

def get_sigma(target_delta, p_, eps, n, sensitivity):
    if np.isinf(eps):
        return 0.0
    
    p = p_/(1-(1-p_)**n)
    eps_p = np.log(1+(1/p)*(np.exp(eps)-1))

    delta = lambda s: p*(norm.cdf(sensitivity/(2*s)-(eps_p*s)/sensitivity) - np.exp(eps_p)*norm.cdf(-sensitivity/(2*s)-(eps_p*s)/sensitivity))

    sigma = minimize(lambda s: np.abs(delta(s) - target_delta), 1.0, method="Nelder-Mead", tol=1e-6).x[0]
   
    return sigma

class BaseProjection:
    
    def __init__(self, epsilon, num_classes, num_dims, participation_probability=1.0, delta=1e-6, sensitivity=np.sqrt(2), normalizer="min_power") -> None:
        self.epsilon = epsilon
        self.participation_probability = participation_probability
        self.delta = delta
        self.sensitivity = sensitivity
        self.num_dims = num_dims
        self.num_classes = num_classes
        self.normalizer = normalizer
        
        self.sigmas = { }
    
    def normalize(self, x):
        if self.normalizer is None:
            return x
        
        return getattr(self, f"{self.normalizer}_normalizer")(x)
    
    def normalize_inverse(self, x):
        if self.normalizer is None:
            return x
        
        return getattr(self, f"{self.normalizer}_normalizer_inverse")(x)
        
    def forward(self, x) -> torch.Tensor:
        raise NotImplementedError
    
    def project_only(self, x) -> torch.Tensor:
        
        return x @ self.W
    
    def invert(self, x) -> torch.Tensor:
        raise NotImplementedError

    def min_power_normalizer(self, x):
        
        return x - 1 / x.shape[1]
    
    def min_power_normalizer_inverse(self, x):
        
        return x + 1 / x.shape[1]
    
    def get_sigma_per_device(self, num_participating_devices, num_devices, sensitivity):
        
        key = (num_participating_devices, num_devices)
        
        if key not in self.sigmas:
            self.sigmas[key] = get_sigma(1e-6, self.participation_probability, self.epsilon, num_devices, sensitivity=sensitivity) / np.sqrt(num_participating_devices)
        
        sigma = self.sigmas[key]
        
        return sigma

class PrivateIdentity(BaseProjection):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self.num_classes == self.num_dims
        
        self.W = torch.eye(self.num_classes)
    
    def forward(self, x, num_participating_devices, num_devices):
        x = self.normalize(x)
        
        sigma = self.get_sigma_per_device(num_participating_devices, num_devices, self.sensitivity)
        
        x = x + torch.randn_like(x) * sigma

        return x
    
    def invert(self, x):
        x = self.normalize_inverse(x)
        
        return x
    
    def get_sigma_client(self, num_participating_devices, num_devices):
        sigma = self.get_sigma_per_device(num_participating_devices, num_devices, self.sensitivity)
        
        return sigma
    
class PrivateGaussianProjection(BaseProjection):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.W = torch.randn((self.num_classes, self.num_dims)) / torch.sqrt(torch.tensor(self.num_dims))
        self.W_inv = torch.pinverse(self.W)
        
        self.W_sensitivity = self.get_sensitivity()
        
    def get_sensitivity(self):
        
        max_dist = torch.tensor(0.0)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                max_dist = torch.maximum(torch.norm(self.W[i] - self.W[j], p=2), max_dist)
        
        return max_dist
    
    def forward(self, x, num_participating_devices, num_devices):
        x = self.normalize(x)
        
        sigma = self.get_sigma_per_device(num_participating_devices, num_devices, self.W_sensitivity)
        
        x = x @ self.W + torch.randn((x.shape[0], self.W.shape[1]), dtype=x.dtype) * sigma
        
        return x
    
    def invert(self, x):
        x = x @ self.W_inv
        
        x = self.normalize_inverse(x)
        return x
    
    def get_sigma_client(self, num_participating_devices, num_devices):
        sigma = self.get_sigma_per_device(num_participating_devices, num_devices, self.W_sensitivity)
        
        return sigma
    
class PrivateGaussianProjectionOfLabels(PrivateGaussianProjection):
    
    def forward(self, x, num_participating_devices, num_devices):
        x = self.normalize(x)
        
        sigma = self.get_sigma_per_device(num_participating_devices, num_devices, self.sensitivity)
        
        x = x + torch.randn_like(x) * sigma
        
        x = x @ self.W

        return x

    def get_sigma_client(self, num_participating_devices, num_devices):
        sigma = self.get_sigma_per_device(num_participating_devices, num_devices, self.sensitivity)
        
        return sigma
    
class PrivateRademacherProjectionOfLabels(PrivateGaussianProjectionOfLabels):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.W = torch.randint(0, 2, (self.num_classes, self.num_dims), dtype=torch.float32)
        self.W[self.W == 0] = -1
        
        self.W_inv = torch.pinverse(self.W)


class PrivateRademacherProjection(PrivateGaussianProjection):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.W = torch.randint(0, 2, (self.num_classes, self.num_dims), dtype=torch.float32)
        self.W[self.W == 0] = -1
        
        self.W_inv = torch.pinverse(self.W)
        
        #self.W_sensitivity = self.sensitivity * torch.sqrt(torch.tensor(self.num_dims))
        self.W_sensitivity = self.get_sensitivity()
        

class PrivateOrthogonalProjection(PrivateGaussianProjection):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        rows = self.W.size(0)
        cols = self.W.size(1)

        if rows < cols:
            self.W.t_()

        # Compute the qr factorization
        self.W, r = torch.linalg.qr(self.W)
        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        d = torch.diag(r, 0)
        ph = d.sign()
        self.W *= ph

        if rows < cols:
            self.W.t_()
        
        self.W_inv = self.W.T
        
        self.W_sensitivity = self.get_sensitivity()

class PrivateOrthogonalProjectionOfLabels(PrivateGaussianProjectionOfLabels):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.W = torch.linalg.qr(self.W).Q
        
        self.W_inv = self.W.T


class PrivateOrthogonalProjectionOfLabelsRandomizedResponse(BaseProjection):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.W = torch.randn((self.num_classes, self.num_dims)) / torch.sqrt(torch.tensor(self.num_dims))

        self.W = torch.linalg.qr(self.W).Q
        
        self.W_inv = self.W.T
    
    def forward(self, x, num_participating_devices, num_devices):
        true_val = torch.argmax(x, dim=1)
        
        for i in range(x.shape[0]):
            chosen_ind = randomize(self.epsilon, self.num_classes, true_val[i])
            #print(x.shape, true_val[i], chosen_ind, self.epsilon, self.num_classes)
            x[i, :] = torch.nn.functional.one_hot(torch.tensor(chosen_ind), self.num_classes)

        x = self.normalize(x)
                
        x = x @ self.W

        return x
    
    def invert(self, x):
        x = x @ self.W_inv
        
        x = self.normalize_inverse(x)
        return x
