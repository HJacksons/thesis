from torchvision import models
import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.models.inception import Inception_V3_Weights


# lets define inception model
class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()

        weights = Inception_V3_Weights.DEFAULT
        self.model = models.inception_v3(weights=weights)
        self.model.aux_logits = False
        self.model.fc = nn.Linear(2048, 3)  # Fine turning the model for 3 classes

    def forward(self, x):
        return self.model(x)


# Print model summary
model = Inception()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
summary(model, input_size=(3, 299, 299))
print(model)
