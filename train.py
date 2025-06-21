import argparse
import logging
import math
import os
import shutil
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm.auto import tqdm

import tensorboard
from accelerate.logging import get_logger
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from omegaconf import OmegaConf

import diffusers
from diffusers import DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from leaf import (
    AutoencoderKL,
    LatentEncoder,
    UNetModel,
    UNetModelWrapper,
    LeafPipeline,
    LeafOutput
)
from src.data import load_custom_dataset
from src.util.loss import cosine_loss
from src.util.metric import SegmentationMetric, Visualization
from src.util.seeding import generate_seed_sequence

logger = get_logger(__name__, log_level="INFO")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train latent diffusion model with VAE and UNet"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML config file"
    )
    return parser.parse_args()

def setup_directories(base_output: str, job_name: str) -> Dict[str, str]:
    out_run = os.path.join(base_output, job_name)
    dirs = {
        "run": out_run,
        "ckpt": os.path.join(out_run, "checkpoint"),
        "tb": os.path.join(out_run, "tensorboard"),
        "vis": os.path.join(out_run, "visualization"),
    }
    return dirs

def replace_unet_input(unet: UNetModel):
    _n_convin_out_channel = unet.input_blocks[0][0].out_channels
    _new_conv_in = nn.Conv2d(
        8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    )
    _weight = unet.input_blocks[0][0].weight.clone()
    _bias = unet.input_blocks[0][0].bias.clone()
    _weight = _weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)
    # half the activation magnitude
    _weight *= 0.5
    _new_conv_in.weight = nn.Parameter(_weight)
    _new_conv_in.bias = nn.Parameter(_bias)
    unet.input_blocks[0][0] = _new_conv_in
    unet.register_to_config(in_channels=8)
    logger.info("Unet config is updated")
    return

@torch.no_grad()
def load_dinov2(model_name: str, resolution: int = 256) -> nn.Module:
    import timm
    encoder = torch.hub.load('facebookresearch/dinov2', model_name)
    del encoder.head  # Remove classification head
    patch_resolution = 16 * (resolution // 256)
    encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
        encoder.pos_embed.data, [patch_resolution, patch_resolution]
    )
    encoder.head = torch.nn.Identity()  # Replace head with identity
    encoder.eval()
    return encoder

@torch.no_grad()
def log_validation(
    vae: AutoencoderKL,
    latent_encoder: AutoencoderKL,
    unet: UNetModel,
    noise_scheduler: DDIMScheduler,
    valid_dataloader: DataLoader,
    cfg: OmegaConf,
    accelerator: Accelerator,
    save_directory: str,
    weight_dtype: torch.dtype,
    step: int
):
    pipeline = LeafPipeline(
        vae=accelerator.unwrap_model(vae),
        unet=accelerator.unwrap_model(unet),
        latent_encoder=accelerator.unwrap_model(latent_encoder),
        scheduler=noise_scheduler,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    pipeline.unet.eval()
    pipeline.latent_encoder.eval()

    val_seed_ls = generate_seed_sequence(cfg.seed, len(valid_dataloader))
    
    metrics = SegmentationMetric(metrics=cfg.metrics, device=accelerator.device)
    visualization = Visualization()

    with torch.autocast(accelerator.device.type):
        for batch in tqdm(
            valid_dataloader,
            desc="Validating",
            total=len(valid_dataloader),
            # disable=not accelerator.is_local_main_process
        ):
            # Read input image (tensor)
            rgb = batch["pixel_values"]  # [B, 3, H, W]
            # GT mask
            mask_gt = batch["mask_values"].to(accelerator.device)

            # Random number generator
            seed = val_seed_ls.pop()
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=accelerator.device)
                generator.manual_seed(seed)
            
            # Predict mask
            with torch.autocast(accelerator.device.type):
                pipe_out: LeafOutput = pipeline(
                    rgb, # don't need to norm outside
                    num_inference_steps=1,
                    generator=generator,
                    show_progress_bar=False,
                )

            mask_pred = pipe_out.mask_pred
            mask_pred_onehot = torch.where(torch.mean(mask_pred, dim=1, keepdim=True) > 0.5, 1, 0).long()
            mask_gt_onehot = torch.mean(mask_gt, dim=1, keepdim=True).long()
            metrics.update(mask_pred_onehot, mask_gt_onehot)

            visualization.update(rgb.cpu(), mask_pred_onehot.cpu(), mask_gt_onehot.cpu())

    # grid = visualization.sample(nrow=cfg.nrow)
    # images = Image.fromarray((grid * 255).astype(np.uint8))
    # images.save(os.path.join(save_directory, f"step-{step:06d}.jpg"))

    results = metrics.compute()
    accelerator.log(results, step=step)

    del pipeline
    # torch.cuda.empty_cache()
    return results

def main():

    args = parse_args()
    cfg = OmegaConf.load(args.config)

    # Create output directories
    dirs = setup_directories(cfg.output_dir, cfg.job_name)

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=cfg.mixed_precision,
        log_with="tensorboard",
        project_config=ProjectConfiguration(project_dir=dirs['run'], logging_dir=dirs['tb'])
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if accelerator.is_main_process:
        for path in dirs.values():
            os.makedirs(path, exist_ok=False)
        
        accelerator.init_trackers("tensorboard")

    # Logging configurations
    log_file = os.path.join(dirs['run'], "logging.log")
    log_format = logging.Formatter('%(asctime)s - %(levelname)s -%(filename)s - %(funcName)s >> %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.INFO)
    logger.logger.addHandler(file_handler)

    logger.info(cfg, main_process_only=True)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    # -------------------- Device --------------------
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # -------------------- Model --------------------
    unet = UNetModel.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="unet", local_files_only=True
    )
    replace_unet_input(unet)
    unet_wrapper = UNetModelWrapper(unet, cfg.use_alignment)

    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="vae", local_files_only=True
    )

    latent_encoder = LatentEncoder()
    latent_encoder.init_from_pretrained(vae)

    noise_scheduler: DDIMScheduler = DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.0015, beta_end=0.0155, prediction_type=cfg.prediction_type
    )
    if cfg.use_alignment:
        vision_encoder = load_dinov2(model_name=cfg.vision_encoder_model)
        vision_encoder.requires_grad_(False)
    
    vae.requires_grad_(False)
    latent_encoder.requires_grad_(True)
    unet_wrapper.train()
    latent_encoder.train()

    # Create EMA for the unet.
    if cfg.use_ema:
        ema_unet = deepcopy(unet)
        ema_unet: EMAModel = EMAModel(
            ema_unet.parameters(),
            model_cls=UNetModel,
            model_config=ema_unet.config,
            foreach=True,
        )

    # For mixed precision training, cast non-trainable weights to half-precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if cfg.use_alignment:
        vision_encoder.to(accelerator.device, dtype=weight_dtype)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            if cfg.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                if isinstance(model, UNetModelWrapper):
                    model.unet.save_pretrained(os.path.join(output_dir, "unet"))
                elif isinstance(model, LatentEncoder):
                    model.save_pretrained(os.path.join(output_dir, "latent_encoder"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    accelerator.register_save_state_pre_hook(save_model_hook)

    # Optimizer and learning rate scheduler
    def model_parameters():
        return list(unet_wrapper.parameters()) + list(latent_encoder.parameters())
    optimizer = optim.AdamW(model_parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    lr_scheduler = get_scheduler(
        name="constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=cfg.num_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.max_train_steps * accelerator.num_processes
    )

    # -------------------- Data --------------------
    logger.info(f"Loading Dataset {cfg.dataset_name}...")
    train_dataset, test_dataset = load_custom_dataset(
        base_data_dir=cfg.base_data_dir,
        dataset_name=cfg.dataset_name,
        resolution=cfg.resolution,
        seed=cfg.seed,
    )

    imagenet_transforms = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        mask_values = torch.stack([example["mask_values"] for example in examples])
        mask_values = mask_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values, "mask_values": mask_values}

    train_dataloader = DataLoader(
        train_dataset,
        cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True
    )

    # Prepare everything with our `accelerator`.
    unet_wrapper, latent_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet_wrapper, latent_encoder, optimizer, train_dataloader, lr_scheduler
    )

    if cfg.use_ema:
        ema_unet.to(accelerator.device)

    # -------------------- Training --------------------
    logger.info("Start Training...")
    global_seed_sequence = generate_seed_sequence(
        initial_seed=cfg.seed,
        length=cfg.max_train_steps,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / accelerator.gradient_accumulation_steps
    )
    max_epoch = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    progress_bar = tqdm(
        range(0, cfg.max_train_steps),
        initial=0,
        desc="Training Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    scaling_factor = 0.18215
    best_dice_score = 0
    best_ckpt_step = 0

    for epoch in range(max_epoch):
        unet_wrapper.train()
        latent_encoder.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):

            with accelerator.accumulate(unet_wrapper):

                # globally consistent random generators
                if cfg.seed is not None:
                    local_seed = global_seed_sequence.pop()
                    rand_num_generator = torch.Generator(device=accelerator.device)
                    rand_num_generator.manual_seed(local_seed)
                else:
                    rand_num_generator = None

                rgb: torch.Tensor = batch["pixel_values"]
                mask: torch.Tensor = batch["mask_values"]

                rgb_norm = rgb * 2.0 - 1.0 # [0, 1] -> [-1, 1]
                mask_norm = mask * 2.0 - 1.0

                # accelerator format code
                rgb_latent = latent_encoder(rgb_norm).mode() # [B, 4, h, w]
                rgb_latent = rgb_latent * scaling_factor

                with torch.no_grad():
                    gt_mask_latent = vae.encode(mask_norm.to(weight_dtype)).mode()
                    gt_mask_latent = gt_mask_latent * scaling_factor

                batch_size = rgb.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    1000,
                    (batch_size,),
                    device=accelerator.device,
                    generator=rand_num_generator,
                ).long()  # [B]

                # Sample noise
                noise = torch.randn(
                    gt_mask_latent.shape,
                    device=accelerator.device,
                    generator=rand_num_generator,
                )  # [B, 4, h, w]

                # Add noise to the latents (diffusion forward process)
                noisy_latents = noise_scheduler.add_noise(
                    gt_mask_latent, noise, timesteps
                )  # [B, 4, h, w]

                # Concat rgb and depth latents
                cat_latents = torch.cat(
                    [rgb_latent, noisy_latents], dim=1
                )  # [B, 8, h, w]
                cat_latents = cat_latents.float()

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "sample":
                    target = gt_mask_latent
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(gt_mask_latent, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    
                # Predict the noise residual and compute loss
                model_pred, z_tilde = unet_wrapper(cat_latents, timesteps)  # [B, 4, h, w]

                loss = F.l1_loss(model_pred.float(), target.float())

                if cfg.use_alignment:
                    with torch.no_grad():
                        rgb_for_dino = F.interpolate(rgb, 224 * (cfg.resolution // 256), mode="bicubic")
                        rgb_for_dino = imagenet_transforms(rgb_for_dino)
                        z: torch.Tensor = vision_encoder.forward_features(rgb_for_dino.to(weight_dtype))["x_norm_patchtokens"]
                    distill_loss = cosine_loss(z.float(), z_tilde.float())
                    loss += cfg.lam * distill_loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.train_batch_size)).mean()
                train_loss += avg_loss.item() / cfg.gradient_accumulation_steps
            
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model_parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if cfg.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                train_loss = 0.0

                if global_step >= cfg.validation_start_steps:
                    
                    if accelerator.is_main_process:
                        if global_step % cfg.validation_steps == 0:
                            if cfg.use_ema:
                                ema_unet.store(unet.parameters())
                                ema_unet.copy_to(unet.parameters())
                            
                            metrics = log_validation(
                                vae,
                                latent_encoder,
                                unet,
                                noise_scheduler,
                                test_dataloader,
                                cfg,
                                accelerator,
                                dirs['vis'],
                                weight_dtype,
                                global_step
                            )
                            if cfg.use_ema:
                                ema_unet.restore(unet.parameters())
                            
                            if best_dice_score < metrics["dice"]:
                                # deleta last best checkpoint
                                last_best_ckpt_path = os.path.join(dirs["ckpt"], f"step-{best_ckpt_step}")
                                best_dice_score = metrics["dice"]
                                best_ckpt_step = global_step

                                if accelerator.is_main_process:
                                    if os.path.exists(last_best_ckpt_path):
                                        shutil.rmtree(last_best_ckpt_path)
                                    save_path = os.path.join(dirs["ckpt"], f"step-{global_step}")
                                    accelerator.save_state(save_path)
                                    logger.info(f"Best Dice Score at step {global_step}: {best_dice_score:.4f}")
                                    logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.max_train_steps:
                break

    # Save the final UNet checkpoint
    accelerator.wait_for_everyone()
    accelerator.end_training()

    logger.info(f"Best Dice Score: {best_dice_score:.4f}")

if __name__ == "__main__":
    main()