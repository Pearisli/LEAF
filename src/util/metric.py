import torch
import pandas as pd
import torchvision.utils as vutils
from torch import Tensor
from monai.metrics import compute_generalized_dice, compute_iou

# Adapted from: https://github.com/victoresque/pytorch-template/blob/master/utils/util.py
class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        # self._data.loc[key, "total"] += value * n
        self._data.loc[key, "total"] += value
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
    
def dice(output: Tensor, target: Tensor) -> Tensor:
    return compute_generalized_dice(output, target, weight_type="uniform")

def iou(output: Tensor, target: Tensor) -> Tensor:
    return compute_iou(output, target, ignore_empty=False)
    
class ImageLogger:
    
    def __init__(self, nrow: int = 12) -> None:
        self.nrow = nrow
        self.reset()
    
    def reset(self) -> None:
        self.v_images = torch.tensor([])
        self.v_segs = torch.tensor([])
        self.v_segs_pred = torch.tensor([])
    
    def add_images(self, images: Tensor, segs: Tensor, segs_pred: Tensor) -> None:
        if segs.shape[1] == 1:
            segs = segs.repeat(1, 3, 1, 1)
        if segs_pred.shape[1] == 1:
            segs_pred = segs_pred.repeat(1, 3, 1, 1)
        self.v_images = torch.cat([self.v_images, images.float()], dim=0)
        self.v_segs = torch.cat([self.v_segs, segs.float()], dim=0)
        self.v_segs_pred = torch.cat([self.v_segs_pred, segs_pred.float()], dim=0)
    
    def _rescale(self, images: Tensor) -> Tensor:
        min_vals = images.view(images.shape[0], -1).min(dim=1)[0].view(-1, 1, 1, 1)
        max_vals = images.view(images.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1)
    
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        return (images - min_vals) / range_vals
    
    def make_grid(self, padding: int = 3, rescale: bool = False):
        def add_padding(image: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.pad(image, (padding, padding, padding, padding), mode="constant", value=1)
        
        target_tensors = [add_padding(tensor) for tensor in [self.v_images, self.v_segs, self.v_segs_pred]]
        if rescale:
            target_tensors = [self._rescale(tensor) for tensor in [self.v_images, self.v_segs, self.v_segs_pred]]
        if target_tensors[0].shape[1] == 4:
            target_tensors = [tensor[:, :3, :, :] for tensor in [self.v_images, self.v_segs, self.v_segs_pred]]

        units = [item for i in range(self.nrow) for item in (target_tensors[0][i], target_tensors[1][i], target_tensors[2][i])]

        grid = vutils.make_grid(units, nrow=self.nrow, normalize=False, padding=3)

        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        return grid_np