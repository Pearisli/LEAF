import os
from .base_seg_dataset import BaseSegDataset
from .cvc import CVC
from .refuge2 import REFUGE2
from .isic import ISIC
from .qata import QaTaCOV

dataset_name_class_dict: dict[str, type[BaseSegDataset]] = {
    "CVC": CVC,
    "REFUGE2": REFUGE2,
    "ISIC18": ISIC,
    "QaTa-COV19": QaTaCOV,
}

def get_dataset(
    base_data_dir: str,
    dataset_name: str,
    resolution: int = 256,
    seed:int = 42,
    **kwargs
) -> dict:

    dataset_class = dataset_name_class_dict[dataset_name]
    dataset_dir = os.path.join(base_data_dir, dataset_name)
    return {
        "train": dataset_class(dataset_dir=dataset_dir, split="train", resolution=resolution, seed=seed, **kwargs),
        "val": dataset_class(dataset_dir=dataset_dir, split="val", resolution=resolution, seed=seed, **kwargs),
        "test": dataset_class(dataset_dir=dataset_dir, split="test", resolution=resolution, seed=seed, **kwargs),
        "pred_to_onehot": dataset_class.pred_to_onehot,
        "gt_to_onehot": dataset_class.gt_to_onehot,
    }