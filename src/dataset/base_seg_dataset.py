import os
import random
from PIL import Image

import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

class BaseSegDataset(Dataset):

    def __init__(
        self,
        dataset_dir: str,
        split: str,
        rgb_subfolder: str,
        mask_subfolder: str,
        resolution: int = 256,
        rgb_name_mode: str = ".jpg",
        mask_name_mode: str = ".png",
        train_index: int = None,
        valid_index: int = None,
        scale_transform=lambda x: x / 255.0,
        seed: int = 42
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        assert os.path.exists(
            self.dataset_dir
        ), f"Dataset does not exist at: {self.dataset_dir}"
        self.split = split

        self.rgb_subfolder = rgb_subfolder
        self.mask_subfolder = mask_subfolder
        self.resolution = resolution
        self.rgb_name_mode = rgb_name_mode
        self.mask_name_mode = mask_name_mode

        # Load base filenames
        base_filenames = os.listdir(os.path.join(self.dataset_dir, self.rgb_subfolder))
        base_filenames = [os.path.splitext(os.path.basename(file))[0] for file in base_filenames if file.endswith(self.rgb_name_mode)]

        self.train_index = train_index
        self.valid_index = valid_index
        self.scale_transform = scale_transform
        self.seed = seed

        # Split data if not specific train/valid partition
        if self.train_index is not None:
            assert split in ("train", "val", "test"), f"Invalid split mode {self.split}"
            base_filenames = self._split_data(base_filenames)
        
        # Load rgb and mask filenames
        self.rgb_filenames = [os.path.join(dataset_dir, rgb_subfolder, file) + self.rgb_name_mode for file in base_filenames]
        self.mask_filenames = [os.path.join(dataset_dir, mask_subfolder, file) + self.mask_name_mode for file in base_filenames]

    def __len__(self):
        return len(self.rgb_filenames)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rgb = self._read_rgb_file(self.rgb_filenames[index])
        mask = self._read_mask_file(self.mask_filenames[index])
        mask = self._get_valid_mask(mask)
        if self.split == "train":
            rgb, mask = self._shape_aug(rgb, mask)
        rgb, mask = self.scale_transform(rgb), self.scale_transform(mask) # [0, 255] -> [0, 1]
        return {
            "rgb": rgb,
            "mask": mask
        }
    
    def _split_data(self, base_filenames: list[str]) -> list[str]:
        random.seed(self.seed)
        if self.split == "train":
            return base_filenames[:self.train_index]
        elif self.split == "valid":
            return base_filenames[self.train_index: self.valid_index]
        else:
            return base_filenames[self.valid_index:]

    def _read_rgb_file(self, rgb_path: str, interpolation: InterpolationMode = InterpolationMode.BILINEAR) -> torch.Tensor:
        rgb = Image.open(rgb_path).convert("RGB")
        rgb = F.resize(rgb, [self.resolution, self.resolution], interpolation)
        rgb = F.pil_to_tensor(rgb)
        return rgb

    def _read_mask_file(self, mask_path: str) -> torch.Tensor:
        return self._read_rgb_file(mask_path, InterpolationMode.NEAREST_EXACT)

    def _get_valid_mask(self, mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask > 128, 255, 0)

    def _shape_aug(self, rgb: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() > 0.5:
            rgb, mask = F.hflip(rgb), F.hflip(mask)
        if random.random() > 0.5:
            rgb, mask = F.vflip(rgb), F.vflip(mask)
        return rgb, mask

    @classmethod
    def pred_to_onehot(cls, mask_pred: torch.Tensor) -> torch.Tensor:
        mask_pred = torch.mean(mask_pred, dim=1, keepdim=True)
        return torch.where(mask_pred > 0.5, 1.0, 0.0).long()

    @classmethod
    def gt_to_onehot(cls, mask_gt: torch.Tensor) -> torch.Tensor:
        mask_gt = torch.mean(mask_gt, dim=1, keepdim=True)
        return mask_gt.long()