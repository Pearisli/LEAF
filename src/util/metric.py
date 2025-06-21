import torch
from typing import List, Dict
from torchvision.utils import make_grid
from torchmetrics.metric import Metric
from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU

class SegmentationMetric:

    def __init__(self, metrics: List[str], device: torch.device):
        self.metrics = metrics
        self.accumulator: Dict[str, Metric] = {
            "dice": GeneralizedDiceScore(num_classes=2, weight_type="linear").to(device),
            "miou": MeanIoU(num_classes=2).to(device)
        }

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        for metric_name in self.metrics:
            self.accumulator[metric_name].update(output, target)
            
    def compute(self) -> Dict[str, float]:
        results = {}
        for metric_name in self.metrics:
            results[metric_name] = self.accumulator[metric_name].compute().item()
        return results

class Visualization:

    def __init__(self):
        self.images = torch.tensor([])
        self.outputs = torch.tensor([])
        self.targets = torch.tensor([])

    def update(self, images: torch.Tensor, output: torch.Tensor, target: torch.Tensor) -> None:
        output = output.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)
        self.images = torch.cat([self.images, images.float()], dim=0)
        self.outputs = torch.cat([self.outputs, output.float()], dim=0)
        self.targets = torch.cat([self.targets, target.float()], dim=0)
    
    def sample(self, nrow: int, padding: int = 3):
        def add_padding(image: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.pad(image, (padding, padding, padding, padding), mode="constant", value=1)
        
        target_tensors = [add_padding(tensor) for tensor in [self.images, self.targets, self.outputs]]
        units = [item for i in range(nrow) for item in (target_tensors[0][i], target_tensors[1][i], target_tensors[2][i])]
        grid = make_grid(units, nrow=nrow, normalize=False, padding=3)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        return grid_np