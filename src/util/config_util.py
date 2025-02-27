import argparse
import yaml
from dataclasses import dataclass, field, asdict

@dataclass
class DataConfig:

    base_data_dir: str = "./assets/datafiles"
    dataset_name: str = "CVC"
    channels: int = 1
    dataloader_num_workers: int = 8
    resize: int = 256
    train_batch_size: int = 4
    valid_batch_size: int = 64
    validation_epochs: int = 200
    valid_start_iter: int = 10000
    scale_factor: float = 0.18215

@dataclass
class ExperimentConfig:

    # Training
    max_train_steps: int = 100000
    num_train_epochs: int = 100
    lr_warmup_steps: int = 10000
    learning_rate: float = 4e-5

    # Optimizer
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

@dataclass
class AcceleratorConfig:

    seed: int = 1337
    allow_tf32: bool = True
    tracker_project_name: str = "rgb2mask"
    output_dir: str = "outputs"
    mixed_precision: str = "no"
    report_to: str = "tensorboard"
    logger: str = "tensorboard"
    add_datetime_prefix: bool = False
    gradient_accumulation_steps: int = 1

@dataclass
class ModelConfig:

    # diffusion pipeline
    vae_path: str = "./assets/weights/vae.pth"
    unet_path: str = "./assets/weights/unet.pth"
    init_type: str = "half"
    enable_xformers: bool = True
    use_ema: bool = True
    dropout: float = 0.2

    # scheduler
    prediction_type: str = "sample"
    num_inference_steps: int = 1
    num_train_timesteps: int = 1000
    
    # reg encoder
    reg_encoder: str = "dinov2_vitb14_reg"
    reg_repr: bool = True
    reg_lam: float = 1.0
    z_dim: int = 768

    # dice decoder
    enable_decoder: bool = False
    pretrained_decoder: bool = False
    dice_lam: float = 1.0

@dataclass
class ProjectConfig:

    data: DataConfig = field(default_factory=DataConfig)
    exp: ExperimentConfig = field(default_factory=ExperimentConfig)
    accelerator: AcceleratorConfig = field(default_factory=AcceleratorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a dataclass to YAML format.")
    parser.add_argument("--output", type=str, help="Output YAML file path")

    args = parser.parse_args()

    config = ProjectConfig()
    config_dict = asdict(config)
    config_yaml = yaml.dump(config_dict, sort_keys=False, allow_unicode=True)

    with open(args.output, "w", encoding="utf-8") as file:
        file.write(config_yaml)
    print(f"Yaml file is saved to {args.output}")