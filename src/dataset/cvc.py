from .base_seg_dataset import BaseSegDataset

class CVC(BaseSegDataset):

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            rgb_subfolder="PNG/Original",
            mask_subfolder="PNG/Ground Truth",
            rgb_name_mode=".png",
            train_index=490,
            valid_index=552,
            **kwargs
        )