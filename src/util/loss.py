import torch
import torch.nn.functional as F
from torch import Tensor

def mean_flat(x):
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def cosine_loss(z: Tensor, z_tilde: Tensor) -> Tensor:
    assert z.shape == z_tilde.shape, f"z and z_tilde must have the same shape, got z: {z.shape} and z_tilde: {z_tilde.shape}"
    z_tilde = F.normalize(z_tilde, dim=-1)
    z = F.normalize(z, dim=-1)
    loss = mean_flat(-(z * z_tilde).sum(dim=-1))
    loss /= z.shape[0]
    return loss