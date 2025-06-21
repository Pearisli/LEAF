import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

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
        seed: int = 42
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        assert os.path.exists(
            self.dataset_dir
        ), f"Dataset does not exist at: {self.dataset_dir}"
        self.split = split

        # Preprocessing the datasets.
        image_transforms = transforms.Compose(
            [
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor()
            ]
        )
        mask_transforms = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), transforms.InterpolationMode.NEAREST_EXACT),
                transforms.PILToTensor(),
                transforms.Lambda(lambda x: torch.where(x > 128, 255.0, 0.0)),
                transforms.Normalize([0.0], [255.0])
            ]
        )

        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms

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
        self.seed = seed

        # Split data if not specific train/valid partition
        if self.train_index is not None:
            assert split in ("train", "val", "test"), f"Invalid split mode {self.split}"
            base_filenames = self._split_data(base_filenames)
        
        # Load rgb and mask filenames
        self.image_filenames = [os.path.join(dataset_dir, rgb_subfolder, file) + self.rgb_name_mode for file in base_filenames]
        self.mask_filenames = [os.path.join(dataset_dir, mask_subfolder, file) + self.mask_name_mode for file in base_filenames]
    
    def _split_data(self, base_filenames: list[str]) -> list[str]:
        random.seed(self.seed)
        if self.split == "train":
            return base_filenames[:self.train_index]
        elif self.split == "valid":
            return base_filenames[self.train_index: self.valid_index]
        else:
            return base_filenames[self.valid_index:]

    def _flip(self, rgb: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() > 0.5:
            rgb, mask = F.hflip(rgb), F.hflip(mask)
        if random.random() > 0.5:
            rgb, mask = F.vflip(rgb), F.vflip(mask)
        return rgb, mask

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index: int):
        image = Image.open(self.image_filenames[index]).convert("RGB")
        mask = Image.open(self.mask_filenames[index]).convert("RGB")
        image = self.image_transforms(image)
        mask = self.mask_transforms(mask)
        if self.split == "train":
            image, mask = self._flip(image, mask)

        return {
            "pixel_values": image,
            "mask_values": mask
        }