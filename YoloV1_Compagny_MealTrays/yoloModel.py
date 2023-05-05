from configparser import ConfigParser

import torch
import torchvision
from torchinfo import summary
from icecream import ic

class YoloModel(torch.nn.Module):
    def __init__(self, S, C, B, pretrained=True):
        super(YoloModel, self).__init__()
        self.S = S
        self.C = C
        self.B = B

        ### Load ResNet model
        PT_weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        model = torchvision.models.efficientnet_b4(weights=PT_weights)
        
        ### Freeze ResNet weights
        for param in model.parameters():
            param.requires_grad = False

        ### Backbone part
        self.backbone = torch.nn.Sequential(*list(model.children())[:-2])

        ### Head part
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(1792, 512, kernel_size=1, padding=0, stride=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.SiLU(),
            torch.nn.Conv2d(512, 256, kernel_size=1, padding=0, stride=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.SiLU(),
        )

        ### Fully connected part
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(0.5),
            torch.nn.LayerNorm(256 * self.S * self.S),
            torch.nn.Linear(256 * self.S * self.S, self.S * self.S * (self.C + self.B*5)),
            # torch.nn.Sigmoid()  # normalized to 0~1 ajouté le 16/11 
        )
    
    def forward(self, input):
        x = self.backbone(input)
        x = self.head(x)
        x = self.fc(x)
        x = x.view(x.size(0), self.S, self.S, self.B * 5 + self.C)
        return x


def yoloModel(load_yoloweights=False, **kwargs) -> YoloModel:    
    config = ConfigParser()
    config.read("config.ini")

    yoloweights = config.get("WEIGHTS", "efficientnet_weights")

    model = YoloModel(**kwargs)
    if load_yoloweights:
        model.load_state_dict(torch.load(yoloweights))
    
    return model
    

if __name__ == "__main__":
    model = yoloModel(S=7, B=2, C=8)
    x = torch.rand(3, 3, 224, 224)
    ic(model(x).shape)
    # summary(model, x.shape)