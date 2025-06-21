import os
from .base_seg_dataset import BaseSegDataset

class QaTaCOV(BaseSegDataset):

    def __init__(
        self,
        **kwargs,
    ) -> None:
        dataset_dir = kwargs.get("dataset_dir")
        split = kwargs.get("split")
        kwargs["dataset_dir"] = os.path.join(dataset_dir, split)
        super().__init__(
            rgb_subfolder="Images",
            mask_subfolder="Ground-truths",
            rgb_name_mode=".png",
            mask_name_mode=".png",
            **kwargs
        )