import argparse
import logging
import os
import shutil
from datetime import datetime
from typing import Dict, Tuple

import torch
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers.schedulers import DDIMScheduler
from omegaconf import OmegaConf

import tensorboard

from leaf.pipeline import LeafPipeline
from leaf.unet import UNetModel
from leaf.autoencoder import AutoencoderKL, AEEncoder
from src.dataset import get_dataset
from src.trainer import LeafTrainer
from src.util.config_util import DataConfig, ProjectConfig
from src.util.logging_util import config_logging

def get_dataloaders(data_cfg: DataConfig, datasets: Dict[str, Dataset]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, valid_dataset, test_dataset = datasets["train"], datasets["val"], datasets["test"]
    return (
        DataLoader(train_dataset, data_cfg.train_batch_size, shuffle=True, num_workers=data_cfg.dataloader_num_workers, pin_memory=True, drop_last=True),
        DataLoader(valid_dataset, data_cfg.valid_batch_size, shuffle=False, num_workers=data_cfg.dataloader_num_workers, pin_memory=True),
        DataLoader(test_dataset,  data_cfg.valid_batch_size, shuffle=False, num_workers=data_cfg.dataloader_num_workers, pin_memory=True),
    )

def main():
    t_start = datetime.now()
    logging.info(f"start at {t_start}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/config.yaml", type=str, help="Path to training config.")
    args = parser.parse_args()

    # -------------------- Initialization --------------------
    cfg = ProjectConfig(**OmegaConf.load(args.config))
    output_dir = cfg.accelerator.output_dir

    # Full job name
    pure_job_name = os.path.basename(args.config).split(".")[0]
    # Add time prefix
    if cfg.accelerator.add_datetime_prefix:
        job_name = f"{t_start.strftime('%y_%m_%d-%H_%M_%S')}-{pure_job_name}"
    else:
        job_name = pure_job_name
    # Output dir
    if output_dir is not None:
        out_dir_run = os.path.join(output_dir, job_name)
    else:
        out_dir_run = os.path.join("./output", job_name)
    os.makedirs(out_dir_run, exist_ok=False)

    # Other directories
    out_dir_ckpt = os.path.join(out_dir_run, "checkpoint")
    if not os.path.exists(out_dir_ckpt):
        os.makedirs(out_dir_ckpt)
    out_dir_tb = os.path.join(out_dir_run, "tensorboard")
    if not os.path.exists(out_dir_tb):
        os.makedirs(out_dir_tb)
    out_dir_eval = os.path.join(out_dir_run, "evaluation")
    if not os.path.exists(out_dir_eval):
        os.makedirs(out_dir_eval)
    out_dir_vis = os.path.join(out_dir_run, "visualization")
    if not os.path.exists(out_dir_vis):
        os.makedirs(out_dir_vis)

    # -------------------- Logging settings --------------------
    logging_cfg = {
        "filename": "logging.log",
        "format": ' %(asctime)s - %(levelname)s -%(filename)s - %(funcName)s >> %(message)s',
        "console_level": 20,
        "file_level": 10,
    }
    config_logging(logging_cfg, out_dir=out_dir_run)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.accelerator.gradient_accumulation_steps,
        mixed_precision=cfg.accelerator.mixed_precision,
        log_with=cfg.accelerator.report_to,
        project_config=ProjectConfiguration(project_dir=out_dir_run, logging_dir=out_dir_tb),
    )
    logging.debug(f"config: {cfg}") 

    # -------------------- Device --------------------

    # Enable TF32 for faster training on Ampere GPUs
    if cfg.accelerator.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # -------------------- Data --------------------
    logging.info(f"Loading dataset {cfg.data.dataset_name}")
    datasets_dict = get_dataset(
        base_data_dir=cfg.data.base_data_dir,
        dataset_name=cfg.data.dataset_name,
        resolution=cfg.data.resize,
        seed=cfg.accelerator.seed,
    )
    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(
        data_cfg=cfg.data, datasets=datasets_dict
    )

    # -------------------- Model --------------------
    # UNet
    unet = UNetModel(
        dropout=cfg.model.dropout,
        num_heads=8,
        use_scale_shift_norm=True,
        resblock_updown=True
    )
    unet.load_state_dict(torch.load(cfg.model.unet_path, weights_only=True))
    logging.info(f"Loading pretrained UNet from {cfg.model.unet_path}")

    # VAE
    vae = AutoencoderKL()
    vae.load_state_dict(torch.load(cfg.model.vae_path, weights_only=True))
    logging.info(f"Loading pretrained VAE from {cfg.model.vae_path}")

    # Noise scheduler
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0155,
        prediction_type=cfg.model.prediction_type
    )

    # Latent encoder
    latent_encoder = AEEncoder()
    latent_encoder.init_from_pretrained(vae)

    model = LeafPipeline(
        unet=unet,
        vae=vae,
        scheduler=noise_scheduler,
        latent_encoder=latent_encoder,
        scale_factor=cfg.data.scale_factor
    )

    # -------------------- Trainer --------------------
    trainer = LeafTrainer(
        cfg=cfg,
        accelerator=accelerator,
        model=model,
        train_dataloader=train_dataloader,
        out_dir_ckpt=out_dir_ckpt,
        out_dir_eval=out_dir_eval,
        out_dir_vis=out_dir_vis,
        valid_dataloader=valid_dataloader,
        test_dataloader=test_dataloader,
        pred_to_onehot=datasets_dict["pred_to_onehot"],
        gt_to_onehot=datasets_dict["gt_to_onehot"],
    )

    # -------------------- Training & Evaluation Loop --------------------
    try:
        trainer.train()
    except Exception as e:
        logging.exception(e)
        logging.info(f"Exception catched, delete current running directory {out_dir_run}")
        shutil.rmtree(out_dir_run)


if __name__ == "__main__":
    main()