import torch.nn as nn
from torchsummary import summary
from torchvision import models


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()

        self.model = models.vgg19(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier[6] = nn.Linear(
            4096, 3
        )  # Fine turning the model for 3 classes

    def forward(self, x):
        return self.model(x)


# Print model summary
model = VGG19()
summary(model, input_size=(3, 256, 256))
# print trainable and non tainable prameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")
