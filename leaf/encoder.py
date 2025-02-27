import torch
from typing import Optional

@torch.no_grad()
def load_repr_encoder(encoder_type: str, resolution: int = 256) -> Optional[torch.nn.Module]:
    if 'dinov2' in encoder_type:
        import timm
        # encoder_name = 'dinov2_vitb14_reg' if 'reg' in encoder_type else 'dinov2_vitb14'
        # encoder = torch.hub.load('facebookresearch/dinov2', encoder_name)
        encoder = torch.hub.load('facebookresearch/dinov2', encoder_type)
        del encoder.head  # Remove classification head
        patch_resolution = 16 * (resolution // 256)
        encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            encoder.pos_embed.data, [patch_resolution, patch_resolution]
        )
        encoder.head = torch.nn.Identity()  # Replace head with identity
        encoder.eval()
        return encoder
    return None