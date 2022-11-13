import torch
import torch.nn as nn

class SampleLoss(nn.Module):
    def __init__(self, reduction=None):
      super(SampleLoss, self).__init__()
      self.reduction = reduction
    
    def forward(self, input, target, sample_weight=None):
        if sample_weight is not None:
            loss = nn.CrossEntropyLoss(reduction='none')(input, target)
            loss = torch.mul(loss, sample_weight).mean(-1)
        else:
            loss = nn.CrossEntropyLoss(reduction="mean")(input, target)
        return loss