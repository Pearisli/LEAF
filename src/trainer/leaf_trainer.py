# An official reimplemented version of Marigold training script.
# Last modified: 2024-08-16
#
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# If you use or adapt this code, please attribute to https://github.com/prs-eth/marigold.
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import logging
import math
import os
import shutil
from contextlib import contextmanager
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from accelerate import Accelerator
from tqdm import tqdm

from diffusers import DDIMScheduler
from diffusers.optimization import get_constant_schedule_with_warmup
from diffusers.training_utils import EMAModel

from leaf.pipeline import LeafPipeline, LeafOutput
from leaf.unet import UNetModel, UNetModelWithReg
from leaf.encoder import load_repr_encoder
from src.util.loss import LossManager
from src.util.config_util import ProjectConfig
from src.util.logging_util import eval_dic_to_text, log_dic
from src.util.metric import MetricTracker, ImageLogger, dice, iou
from src.util.seeding import generate_seed_sequence

class LeafTrainer:
    
    def __init__(
        self,
        cfg: ProjectConfig,
        accelerator: Accelerator,
        model: LeafPipeline,
        train_dataloader: DataLoader,
        out_dir_ckpt: str,
        out_dir_eval: str,
        out_dir_vis: str,
        valid_dataloader: DataLoader = None,
        test_dataloader: DataLoader = None,
        pred_to_onehot: Callable = None,
        gt_to_onehot: Callable = None,
    ):
        self.cfg = cfg
        self.accelerator = accelerator
        self.model = model
        self.device = self.accelerator.device
        self.seed = self.cfg.accelerator.seed
        self.out_dir_ckpt = out_dir_ckpt
        self.out_dir_eval = out_dir_eval
        self.out_dir_vis = out_dir_vis
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        self.pred_to_onehot = pred_to_onehot
        self.gt_to_onehot = gt_to_onehot

        self.regunet = UNetModelWithReg(
            self.model.unet,
            use_reg=self.cfg.model.reg_repr,
            z_dim=self.cfg.model.z_dim
        )

        # representation regularization
        if self.cfg.model.reg_repr:
            logging.info(f"Activate features alignment, encoder type: {self.cfg.model.reg_encoder}")
            self.repr_encoder = load_repr_encoder(self.cfg.model.reg_encoder, self.cfg.data.resize)
            self.repr_encoder.requires_grad_(False)

        # Adapt input layers
        if self.model.unet.in_channels != 8:
            self._replace_unet_conv_in()
        # ema unet
        ema_unet = UNetModel(
            in_channels=8,
            num_heads=8,
            use_scale_shift_norm=True,
            resblock_updown=True
        )
        self.ema_unet = EMAModel(
            ema_unet.parameters(),
            model_cls=UNetModel,
            foreach=True,
        )
        self.ema_unet.to(self.accelerator.device)

        self.model.enable_xformers_memory_efficient_attention()

        # Trainability
        self.model.vae.requires_grad_(False)
        self.regunet.requires_grad_(True)
        self.model.latent_encoder.requires_grad_(True)

        # Optimizer !should be defined after input layer is adapted
        learning_rate = self.cfg.exp.learning_rate
        self.optimizer = AdamW(
            self._parameters(),
            lr=learning_rate,
            betas=(self.cfg.exp.adam_beta1, self.cfg.exp.adam_beta2),
            weight_decay=self.cfg.exp.adam_weight_decay,
            eps=self.cfg.exp.adam_epsilon
        )

        # constant scheduler
        self.lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.exp.lr_warmup_steps
        )

        # Training noise scheduler
        self.training_noise_scheduler: DDIMScheduler = self.model.scheduler
        self.prediction_type = self.training_noise_scheduler.config.prediction_type
        self.scheduler_timesteps = (
            self.training_noise_scheduler.config.num_train_timesteps
        )
    
        # Eval metrics
        self.metric_funcs = [dice, iou]
        self.train_metrics = MetricTracker(*["loss"])
        self.val_metrics = MetricTracker(*[m.__name__ for m in self.metric_funcs])
        self.image_logger = ImageLogger(nrow=6)
        
        # main metric for best checkpoint saving
        self.main_val_metric = "dice"
        self.best_val_metric = 0.0
        self.best_test_metric = 0.0

        # Settings
        self.dtype = torch.float32
        if self.cfg.accelerator.mixed_precision == "fp16":
            self.dtype = torch.float16
        elif self.cfg.accelerator.mixed_precision == "bf16":
            self.dtype = torch.bfloat16
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.cfg.accelerator.gradient_accumulation_steps)
        self.max_epoch = math.ceil(self.cfg.exp.max_train_steps / num_update_steps_per_epoch)
        self.max_train_steps = self.cfg.exp.max_train_steps
        self.resolution = self.cfg.data.resize

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(self.cfg.accelerator.tracker_project_name)
            self.tracker = self.accelerator.trackers[0] # tensorboard tracker

        # Internal variables
        self.epoch = 1
        self.effective_iter = 0  # how many times optimizer.step() is called
        self.global_seed_sequence: List = []  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming
        self.imagenet_norm = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def _replace_unet_conv_in(self):
        # new conv_in channel
        _n_convin_out_channel = self.model.unet.input_blocks[0][0].out_channels
        _new_conv_in = nn.Conv2d(
            8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _weight = self.model.unet.input_blocks[0][0].weight.clone()
        _bias = self.model.unet.input_blocks[0][0].bias.clone()
        if self.cfg.model.init_type == "half":
            _weight = _weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)
            # half the activation magnitude
            _weight *= 0.5
        elif self.cfg.model.init_type == "zero":
            _weight = torch.cat([torch.zeros_like(_weight), _weight], dim=1)
        _new_conv_in.weight = nn.Parameter(_weight)
        _new_conv_in.bias = nn.Parameter(_bias)
        self.model.unet.input_blocks[0][0] = _new_conv_in
        self.model.unet.in_channels *= 2
        logging.info(f"Unet conv_in layer is replaced, using {self.cfg.model.init_type} initialization")

    def _unwrap(self, model):
        return self.accelerator.unwrap_model(model)

    def _parameters(self, recurse: bool = True):
        params = list(self.regunet.parameters(recurse))
        params += list(self.model.latent_encoder.parameters(recurse))
        return params

    def _prepare_everything(self, device: torch.device, dtype: torch.dtype):
        (self.regunet, self.model.latent_encoder,
         self.train_dataloader, self.valid_dataloader, self.test_dataloader,
         self.optimizer, self.lr_scheduler) = self.accelerator.prepare(
            self.regunet, self.model.latent_encoder,
            self.train_dataloader, self.valid_dataloader, self.test_dataloader,
            self.optimizer, self.lr_scheduler
        )

        self.model.vae.to(device, dtype)
        # self.model.vae.decoder.eval()
        
        if self.cfg.model.reg_repr:
            self.repr_encoder.to(device, dtype)
        
    def _prepare_repr_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(rgb, 224 * (self.resolution // 256), mode="bicubic")
        x = self.imagenet_norm(x)
        return x

    @torch.no_grad()
    def _encode_mask(self, mask_gt_for_latent: torch.Tensor) -> torch.Tensor:
        gt_mask_latent = self.model.vae.encode(mask_gt_for_latent).mode()
        gt_mask_latent = gt_mask_latent * self.model.scale_factor
        return gt_mask_latent

    @contextmanager
    def ema_scope(self):
        if self.cfg.model.use_ema:
            self.ema_unet.store(self.model.unet.parameters())
            self.ema_unet.copy_to(self.model.unet.parameters())
        try:
            yield None
        finally:
            if self.cfg.model.use_ema:
                self.ema_unet.restore(self.model.unet.parameters())

    def train(self):
        logging.info("Start training")

        device, dtype = self.device, self.dtype
        self._prepare_everything(device, dtype)
        
        self.train_metrics.reset()

        progress_bar = tqdm(
            range(self.cfg.exp.max_train_steps),
            desc="Steps",
            disable=not self.accelerator.is_local_main_process,
        )

        for epoch in range(self.max_epoch):
            # Skip previous batches when resume
            for i, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model.unet):
                    self.model.unet.train()
                    self.model.latent_encoder.train()

                    # globally consistent random generators
                    if self.seed is not None:
                        local_seed = self._get_next_seed()
                        rand_num_generator = torch.Generator(device=device)
                        rand_num_generator.manual_seed(local_seed)
                    else:
                        rand_num_generator = None

                    # >>> With gradient accumulation >>>

                    # Get data # (if use accelerator don't need to move to device)
                    rgb: torch.Tensor = batch["rgb"]
                    mask_gt: torch.Tensor = batch["mask"]
                    
                    # Norm data
                    rgb_norm = rgb * 2.0 - 1.0 # [0, 1] -> [-1, 1]
                    mask_gt_for_latent = mask_gt * 2.0 - 1.0

                    batch_size = rgb.shape[0]

                    # accelerator format code
                    rgb_latent = self.model.encode_rgb(rgb_norm) # [B, 4, h, w]
                    gt_mask_latent = self._encode_mask(
                        mask_gt_for_latent.to(dtype)
                    ) # [B, 4, h, w]

                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        self.scheduler_timesteps,
                        (batch_size,),
                        device=device,
                        generator=rand_num_generator,
                    ).long()  # [B]

                    # Sample noise
                    noise = torch.randn(
                        gt_mask_latent.shape,
                        device=device,
                        generator=rand_num_generator,
                    )  # [B, 4, h, w]

                    # Add noise to the latents (diffusion forward process)
                    noisy_latents = self.training_noise_scheduler.add_noise(
                        gt_mask_latent, noise, timesteps
                    )  # [B, 4, h, w]

                    # Concat rgb and depth latents
                    cat_latents = torch.cat(
                        [rgb_latent, noisy_latents], dim=1
                    )  # [B, 8, h, w]
                    cat_latents = cat_latents.float()

                    # Predict the noise residual, reg type code
                    model_pred, z_tilde, last_hidden_states = self.regunet.forward( # to unet wrapper
                        cat_latents, timesteps
                    )  # [B, 4, h, w]

                    if torch.isnan(model_pred).any():
                        logging.warning("model_pred contains NaN.")

                    # Get the target for loss depending on the prediction type
                    if "epsilon" == self.prediction_type:
                        target = noise
                    elif "sample" == self.prediction_type:
                        target = gt_mask_latent
                    elif "v_prediction" == self.prediction_type:
                        target = self.training_noise_scheduler.get_velocity(
                            gt_mask_latent, noise, timesteps
                        )  # [B, 4, h, w]
                    else:
                        raise ValueError(f"Unknown prediction type {self.prediction_type}")

                    # Masked latent loss
                    loss = LossManager.latent_loss(model_pred.float(), target.float())
                    
                    # Representation regularization
                    if self.cfg.model.reg_repr:
                        with torch.no_grad():
                            x = self._prepare_repr_rgb(rgb)
                            z: torch.Tensor = self.repr_encoder.forward_features(x.to(self.dtype))["x_norm_patchtokens"]
                        reg_loss = LossManager.cosine_loss(z.float(), z_tilde.float())
                        loss += self.cfg.model.reg_lam * reg_loss.mean()

                    self.train_metrics.update("loss", loss.item())
                    
                    self.accelerator.backward(loss)

                    # Practical batch end

                    # Clip gradient
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self._parameters(), self.cfg.exp.max_grad_norm)
                    
                    # Perform optimization step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if self.accelerator.sync_gradients:

                    self.ema_unet.step(self.model.unet.parameters()) # EMA model

                    self.effective_iter += 1

                    # Log to tensorboard
                    accumulated_loss = self.train_metrics.result()["loss"]

                    self.train_metrics.reset()

                    progress_bar.update(1)
                    progress_bar.set_postfix(**{
                        "lr": f"{self.lr_scheduler.get_last_lr()[0]:.6f}",
                        "loss": f"{accumulated_loss:.4f}"
                    })

                    # Per-step callback
                    self._train_step_callback()

                    # End of training
                    if self.effective_iter >= self.max_train_steps:
                        self.save_checkpoint(
                            ckpt_name=self._get_backup_ckpt_name(),
                            save_train_state=True,
                        )
                        logging.info("Training ended.")
                        logging.info(f"Best 'Dice' metric on val: {self.best_val_metric}")
                        logging.info(f"Best 'Dice' metric on test: {self.best_test_metric}")

                        # self.validate_single_dataset(self.test_dataloader, metric_tracker=self.val_metrics, prefix="test")
                        return
                # torch.cuda.empty_cache()
                # <<< Effective batch end <<<
            # Epoch end

    def _train_step_callback(self):
        """Executed after every iteration"""
        # Validation
        if 0 == self.effective_iter % self.cfg.data.validation_epochs and self.effective_iter >= self.cfg.data.valid_start_iter:
            # self.save_checkpoint(ckpt_name="latest", save_train_state=True)
            # _is_latest_saved = True

            self.validate(self.valid_dataloader, "val")
            # self.save_checkpoint(ckpt_name="latest", save_train_state=True)

    def validate(self, data_loader: DataLoader, prefix: str):
        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
        with self.ema_scope():

            val_dataset_name = self.cfg.data.dataset_name
            # val_metric_dic = self.validate_single_dataset(
            #     data_loader=data_loader, metric_tracker=self.val_metrics, prefix=prefix
            # )
            # logging.info(
            #     f"Iter {self.effective_iter}. {prefix} metrics on `{val_dataset_name}`: {val_metric_dic}"
            # )

            # log_dic(
            #     self.tracker.writer,
            #     {f"{prefix}/{val_dataset_name}/{k}": v for k, v in val_metric_dic.items()},
            #     global_step=self.effective_iter
            # )

            # # save to file
            # eval_text = eval_dic_to_text(
            #     val_metrics=val_metric_dic,
            #     dataset_name=val_dataset_name,
            # )
            # _save_to = os.path.join(
            #     self.out_dir_eval,
            #     f"eval-{val_dataset_name}-iter{self.effective_iter:06d}.txt",
            # )
            # with open(_save_to, "w+") as f:
            #     f.write(eval_text)

            # Update main eval metric
            if prefix != "train":
                # main_eval_metric = val_metric_dic[self.main_val_metric]
                # if main_eval_metric > self.best_val_metric:
                #     self.best_val_metric = main_eval_metric
                #     logging.info(
                #         f"Best val metric: {self.main_val_metric} = {self.best_val_metric} at iteration {self.effective_iter}"
                #     )
                #     # Save a checkpoint
                #     self.save_checkpoint(
                #         ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
                #     )

                # evaluate test
                test_metric_dic = self.validate_single_dataset(
                    data_loader=self.test_dataloader, metric_tracker=self.val_metrics, prefix="test"
                )
                if test_metric_dic[self.main_val_metric] > self.best_test_metric:
                    self.best_test_metric = test_metric_dic[self.main_val_metric]
                    logging.info(
                        f"Best test metric: {self.main_val_metric} = {self.best_test_metric} at iteration {self.effective_iter}"
                    )
                    self.save_checkpoint(
                        ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
                    )
                    log_dic(
                        self.tracker.writer,
                        {f"test/{val_dataset_name}/{k}": v for k, v in test_metric_dic.items()},
                        global_step=self.effective_iter
                    )

    @torch.no_grad()
    def validate_single_dataset(
        self,
        data_loader: DataLoader,
        metric_tracker: MetricTracker,
        prefix: str,
        save_to_dir: str = None,
    ):
        self.model.unet.eval()
        self.model.latent_encoder.eval()

        metric_tracker.reset()
        # Generate seed sequence for consistent evaluation
        val_seed_ls = generate_seed_sequence(self.seed, len(data_loader)) # Batch size will affect the performance

        val_dataset_name = self.cfg.data.dataset_name

        for i, batch in enumerate(
            tqdm(data_loader, desc=f"evaluating on {prefix} {val_dataset_name}"),
            start=1,
        ):
            # Read input image (tensor)
            rgb = batch["rgb"]  # [B, 3, H, W]
            # GT mask
            mask_gt = batch["mask"].to(self.device)

            # Random number generator
            seed = val_seed_ls.pop()
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)

            # Predict mask
            with torch.autocast(self.device.type):
                pipe_out: LeafOutput = self.model(
                    rgb, # don't need to norm outside
                    num_inference_steps=self.cfg.model.num_inference_steps,
                    generator=generator,
                    show_progress_bar=False,
                )

            mask_pred = pipe_out.mask_pred

            # Evaluate
            for met_func in self.metric_funcs:
                _metric_name = met_func.__name__
                _metric = met_func(self.pred_to_onehot(mask_pred), self.gt_to_onehot(mask_gt))
                metric_tracker.update(_metric_name, _metric.sum().item(), _metric.shape[0])

            self.image_logger.add_images(rgb.cpu(), self.gt_to_onehot(mask_gt).cpu(), self.pred_to_onehot(mask_pred).cpu())

        if 0 == self.effective_iter % 2500: # comment for faster training
            np_images = self.image_logger.make_grid()
            self.tracker.writer.add_images(
                f"vis/{prefix}/step{self.effective_iter}",
                np_images,
                self.effective_iter,
                dataformats="HWC"
            )
        self.image_logger.reset()

        return metric_tracker.result()

    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_train_steps,
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def save_checkpoint(self, ckpt_name: str, save_train_state: bool):
        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.info(f"Saving checkpoint to: {ckpt_dir}")
        # Backup previous checkpoint
        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            temp_ckpt_dir = os.path.join(
                os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
            )
            if os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            os.rename(ckpt_dir, temp_ckpt_dir)
            logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")

        # Save modules
        unet_path = os.path.join(ckpt_dir, "unet")
        self.model.unet.save_pretrained(unet_path, safe_serialization=True)
        logging.info(f"UNet is saved to: {unet_path}")

        encoder_path = os.path.join(ckpt_dir, "latent_encoder")
        self.model.latent_encoder.save_pretrained(encoder_path, safe_serialization=True)
        logging.info(f"Latent encoder is saved to: {encoder_path}")

        scheduler_path = os.path.join(ckpt_dir, "scheduler")
        self.model.scheduler.save_pretrained(scheduler_path)

        if save_train_state:
            state = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "config": self.cfg,
                "effective_iter": self.effective_iter,
                "epoch": self.epoch,
                "best_val_metric": self.best_val_metric,
                "global_seed_sequence": self.global_seed_sequence,
            }
            train_state_path = os.path.join(ckpt_dir, "trainer.ckpt")
            torch.save(state, train_state_path)
            # iteration indicator
            f = open(os.path.join(ckpt_dir, self._get_backup_ckpt_name()), "w")
            f.close()

            logging.info(f"Trainer state is saved to: {train_state_path}")

        # Remove temp ckpt
        if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logging.debug("Old checkpoint backup is removed.")

    def _get_backup_ckpt_name(self):
        return f"iter_{self.effective_iter:06d}"
