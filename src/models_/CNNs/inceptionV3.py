from torchvision import models
import torch
import torch.nn as nn
from torchsummary import summary


# lets define inception model
class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()

        self.model = models.inception_v3(pretrained=True)
        self.model.aux_logits = False
        self.model.fc = nn.Linear(2048, 3)

    def forward(self, x):
        return self.model(x)
