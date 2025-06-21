import os
from .base_seg_dataset import BaseSegDataset
from .cvc import CVC
from .isic import ISIC
from .qata import QaTaCOV

dataset_name_class_dict: dict[str, type[BaseSegDataset]] = {
    "CVC": CVC,
    "ISIC18": ISIC,
    "QaTa-COV19": QaTaCOV,
}

def load_custom_dataset(
    base_data_dir: str,
    dataset_name: str,
    resolution: int = 256,
    seed:int = 42,
    **kwargs
) -> dict:

    data_cls = dataset_name_class_dict[dataset_name]
    dataset_dir = os.path.join(base_data_dir, dataset_name)
    return (
        data_cls(dataset_dir=dataset_dir, split="train", resolution=resolution, seed=seed, **kwargs),
        data_cls(dataset_dir=dataset_dir, split="test",  resolution=resolution, seed=seed, **kwargs),
    )
