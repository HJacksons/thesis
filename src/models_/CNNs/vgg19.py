import torch.nn as nn
from torchsummary import summary
from torchvision import models


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()

        self.model = models.vgg19(pretrained=True)
        self.model.classifier[6] = nn.Linear(
            4096, 3
        )  # Fine turning the model for 3 classes

    def forward(self, x):
        return self.model(x)


# Print model summary
model = VGG19()
summary(model, input_size=(3, 256, 256))
