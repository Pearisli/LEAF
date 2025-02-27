import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Union

from tqdm.auto import tqdm
from diffusers import DiffusionPipeline, DDIMScheduler
from leaf.unet import UNetModel
from leaf.autoencoder import AutoencoderKL, AEEncoder

@dataclass
class LeafOutput:

    mask_pred: torch.Tensor # float format, [0, 1] (B, 3, H, W)
    mask_np: np.ndarray # [0, 1] (B, 3, H, W)

class LeafPipeline(DiffusionPipeline):

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNetModel,
        latent_encoder: AEEncoder,
        scheduler: DDIMScheduler,
        scale_factor: float = 0.18215
    ) -> None:

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            latent_encoder=latent_encoder
        )
        self.scale_factor = scale_factor
    
    def eval(self):
        self.vae.decoder.eval()
        self.unet.eval()
        self.latent_encoder.eval()
    
    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        rgb_latent = self.latent_encoder.encode(rgb_in).mode()
        rgb_latent = rgb_latent * self.scale_factor
        return rgb_latent

    def decode_mask(self, mask_latent: torch.Tensor) -> torch.Tensor:
        mask_latent = mask_latent / self.scale_factor
        stacked = self.vae.decode(mask_latent)
        return stacked

    def single_infer(
        self,
        rgb_norm: torch.Tensor,
        num_inference_steps: int,
        timesteps: List[int] = None,
        generator: Optional[torch.Generator] = None,
        show_pbar: bool = True,
    ) -> torch.Tensor:
        device = self.device
        rgb_norm = rgb_norm.to(device)

        self.scheduler.set_timesteps(num_inference_steps, device)
        if num_inference_steps == 1:
            timesteps = torch.tensor([self.scheduler.config.num_train_timesteps - 1]).to(device).long()
        else:
            timesteps = self.scheduler.timesteps
        
        rgb_latent = self.encode_rgb(rgb_norm)

        # Initial depth map (noise)
        mask_latent = torch.randn(
            rgb_latent.shape,
            device=device,
            dtype=self.dtype,
            generator=generator,
        )  # [B, 4, h, w]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            unet_input = torch.cat(
                [rgb_latent, mask_latent], dim=1
            )  # this order is important

            # predict the noise residual
            timestep = torch.tensor([t]).repeat(rgb_latent.shape[0]).to(device).long()
            model_pred = self.unet(unet_input, timestep).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            if num_inference_steps == 1 and self.scheduler.config.prediction_type != "epsilon":
                mask_latent = self.scheduler.step(
                    model_pred, t, mask_latent, generator=generator
                ).pred_original_sample
            else:
                mask_latent = self.scheduler.step(
                    model_pred, t, mask_latent, generator=generator
                ).prev_sample

        mask = self.decode_mask(mask_latent)

        # clip prediction
        mask = torch.clamp(mask, -1.0, 1.0)
        # shift to [0, 1]
        mask = (mask + 1.0) / 2.0

        return mask

    @torch.no_grad()
    def __call__(
        self,
        rgb_in: torch.Tensor,
        num_inference_steps: int = 50,
        processing_res: int = 256,
        generator: Union[torch.Generator, None] = None,
        resample_method: str = "bilinear",
        show_progress_bar: bool = False
    ) -> LeafOutput:
        rgb_norm: torch.Tensor = rgb_in * 2.0 - 1.0
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # ----------------- Predicting segmentation mask -----------------
        mask_pred = self.single_infer(
            rgb_norm=rgb_norm,
            num_inference_steps=num_inference_steps,
            show_pbar=show_progress_bar,
            generator=generator
        ) # [B, 3, H, W]
        # torch.cuda.empty_cache()

        mask_np = mask_pred.cpu().numpy()

        return LeafOutput(
            mask_pred=mask_pred,
            mask_np=mask_np
        )
    