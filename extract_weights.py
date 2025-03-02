import torch
from typing import Dict
from leaf.autoencoder import AutoencoderKL
from leaf.unet import UNetModel

if __name__ == "__main__":

    print("Loading VAE pre-trained weights")
    vae_config = torch.load("./assets/vae/model.ckpt", weights_only=False)
    vae = AutoencoderKL(dropout=0.2)
    vae.load_state_dict(vae_config["state_dict"], strict=False)
    print("Saving VAE pre-trained weights")
    vae.save_pretrained("./assets/vae")

    print("Loading U-Net pre-trained weights")
    unet_config = torch.load("./assets/unet/model.ckpt", weights_only=False)
    unet_pretrained_state_dict: Dict[str, torch.Tensor] = unet_config["state_dict"]
    unet_state_dict = {}
    unet_params_prefix = "model.diffusion_model."
    for key, item in unet_pretrained_state_dict.items():
        if key.startswith(unet_params_prefix):
            unet_state_dict[key[len(unet_params_prefix):]] = item
    unet = UNetModel(
        dropout=0.2,
        num_heads=8,
        resblock_updown=True,
        use_scale_shift_norm=True
    )
    unet.load_state_dict(unet_state_dict, strict=True)
    print("Saving U-Net pre-trained weights")
    unet.save_pretrained("./assets/unet")