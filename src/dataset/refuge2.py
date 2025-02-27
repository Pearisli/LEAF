import os
import torch
from .base_seg_dataset import BaseSegDataset

class REFUGE2(BaseSegDataset):

    cup_only: bool = True

    def __init__(
        self,
        cup_only: bool = True,
        **kwargs
    ) -> None:
        dataset_dir = kwargs.get("dataset_dir")
        split = kwargs.get("split")
        kwargs["dataset_dir"] = os.path.join(dataset_dir, split)
        super().__init__(
            rgb_subfolder="images",
            mask_subfolder="mask",
            rgb_name_mode=".png",
            mask_name_mode=".png",
            **kwargs
        )
        REFUGE2.cup_only = cup_only
    
    def _get_valid_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if REFUGE2.cup_only:
            mask = mask[1:2, :, :].repeat(3, 1, 1)
        return mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rgb = self._read_rgb_file(self.rgb_filenames[index])
        mask = self._read_mask_file(self.mask_filenames[index])
        mask = self._get_valid_mask(mask)

        if self.split == "train":
            rgb, mask = self._shape_aug(rgb, mask)
        rgb, mask = self.scale_transform(rgb), self.scale_transform(mask)
        return {
            "rgb": rgb,
            "mask": mask
        }
    
    @classmethod
    def pred_to_onehot(cls, mask_pred: torch.Tensor) -> torch.Tensor:
        if REFUGE2.cup_only:
            mask_pred = torch.mean(mask_pred, dim=1, keepdim=True)
            mask_pred = torch.where(mask_pred > 0.5, 1.0, 0.0)
        else:
            mask_pred = torch.mean(mask_pred, dim=1, keepdim=True)
            mask_pred = (255 * mask_pred).long()
            bsz, _, h, w = mask_pred.shape
            mask_pred_onehot = torch.zeros((bsz, 3, h, w)).long()
            mask_pred_onehot[:, 0:1, :, :] = (mask_pred < 127).long()
            mask_pred_onehot[:, 1:2, :, :] = (127 <= mask_pred & mask_pred <= 128).long()
            mask_pred_onehot[:, 2:3, :, :] = (mask_pred > 128).long()
        return mask_pred.long()

    @classmethod
    def gt_to_onehot(cls, mask_gt: torch.Tensor) -> torch.Tensor:
        if REFUGE2.cup_only:
            mask_gt = torch.mean(mask_gt, dim=1, keepdim=True)
        else:
            mask_gt = torch.mean(mask_gt, dim=1, keepdim=True)
            mask_gt = (255 * mask_gt).long()
            bsz, _, h, w = mask_gt.shape
            mask_gt_onehot = torch.zeros((bsz, 3, h, w)).long()
            mask_gt_onehot[:, 0:1, :, :] = (mask_gt == 0).long()
            mask_gt_onehot[:, 1:2, :, :] = (mask_gt == 128).long()
            mask_gt_onehot[:, 2:3, :, :] = (mask_gt == 255).long()
        return mask_gt.long()