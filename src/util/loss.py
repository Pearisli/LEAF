import torch
import torch.nn.functional as F
from torch import Tensor
from monai.losses.dice import DiceCELoss

def mean_flat(x):
    return torch.mean(x, dim=list(range(1, len(x.size()))))

class LossManager:

    @staticmethod
    def latent_loss(latent_output: Tensor, latent_target: Tensor, loss_type: str = "mse") -> Tensor:
        assert loss_type in ["l1", "mse"],  "Invalid latent loss type. Choose 'l1' or 'mse'."
        if loss_type == "mse":
            return F.mse_loss(latent_output, latent_target)
        else:
            return F.l1_loss(latent_output, latent_target)
    
    @staticmethod
    def cosine_loss(z: Tensor, z_tilde: Tensor) -> Tensor:
        assert z.shape == z_tilde.shape, f"z and z_tilde must have the same shape, got z: {z.shape} and z_tilde: {z_tilde.shape}"
        z_tilde = F.normalize(z_tilde, dim=-1)
        z = F.normalize(z, dim=-1)
        loss = mean_flat(-(z * z_tilde).sum(dim=-1))
        loss /= z.shape[0]
        return loss

    @staticmethod
    def semantic_loss(mask_output: Tensor, mask_target: Tensor) -> Tensor:
        return DiceCELoss(sigmoid=True, reduction="mean", include_background=True, lambda_dice=0)(mask_output, mask_target)
        # return F.binary_cross_entropy_with_logits(mask_output, mask_target)
