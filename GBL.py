import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import FocalLoss,DiceLoss
def weight_gaze(heatmap):
    # 提取红色和蓝色通道
    gray_channel = heatmap[:, 0, :, :]
    #green_channel = heatmap[:, 1, :, :]
    # 计算权重，红色通道越大，权重越大；蓝色通道越大，权重越小
    weights = gray_channel
    max_value=torch.max(weights)
    min_value=torch.min(weights)
    if max_value==0 and min_value==0:
        weights=2*torch.ones_like(weights)
        return weights
    else:
        weights = 1 + (weights - min_value) * (239 / (max_value - min_value))
        return weights
class GazeWeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(GazeWeightedCrossEntropyLoss, self).__init__()
        self.FL=FocalLoss()
        #self.FL=nn.CrossEntropyLoss()
    def forward(self, input, target,heatmap):
        # 计算交叉熵损失
        input=input.squeeze(1)
        target=target.squeeze(1)
        weight=weight_gaze(heatmap)
        loss = self.FL(input*weight,target)
        return loss