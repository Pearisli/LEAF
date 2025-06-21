from .base_seg_dataset import BaseSegDataset

class ISIC(BaseSegDataset):

    def __init__(
        self,
        **kwargs
    ) -> None:
        split = kwargs.get("split")
        super().__init__(
            # rgb_subfolder=f"{split}_Data",
            # mask_subfolder=f"{split}_GroundTruth",
            rgb_subfolder="train_Data",
            mask_subfolder="train_GroundTruth",
            mask_name_mode="_segmentation.png",
            train_index=1815,
            # valid_index=2074,
            valid_index=1815,
            **kwargs,
        )