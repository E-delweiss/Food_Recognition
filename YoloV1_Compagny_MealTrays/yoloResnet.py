from configparser import ConfigParser

import torch
import torchvision
from torchinfo import summary

class YoloResNet(torch.nn.Module):
    def __init__(self, S, C, B, pretrained=False):
        super(YoloResNet, self).__init__()
        self.S = S
        self.C = C
        self.B = B

        ### Load ResNet model
        resnet = torchvision.models.resnet152(pretrained=True)
        
        ### Freeze ResNet weights
        for param in resnet.parameters():
            param.requires_grad = False

        ### Backbone part
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-4])

        ### Head part
        self.head = torch.nn.Sequential(
            torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(512, 1024, 3, 1, 1),
            torch.nn.Conv2d(1024, 512, 1, 1, 0),
            torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(512, 1024, 3, 1, 1),
            torch.nn.Conv2d(1024, 512, 1, 1, 0),
            torch.nn.MaxPool2d(2,2)
        )
        ### Fully connected part
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512 * self.S * self.S, 4096), # 4096 -> 496 modifié le 16/11 
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(0.5), # dropout ajouté le 16/11 
            torch.nn.Linear(4096, self.S * self.S * (self.C + self.B*5)),
            torch.nn.Sigmoid()  # normalized to 0~1 ajouté le 16/11 
        )
    
    def forward(self, input):
        x = self.backbone(input)
        x = self.head(x)
        x = self.fc(x)
        x = x.view(x.size(0), self.S, self.S, self.B * 5 + self.C)
        return x


def yoloResnet(load_yoloweights=False, **kwargs) -> YoloResNet:    
    config = ConfigParser()
    config.read("config.ini")

    yoloweights = config.get("WEIGHTS", "resnetYolo_weights")

    model = YoloResNet(**kwargs)
    if load_yoloweights:
        model.load_state_dict(torch.load(yoloweights))
    
    return model
    

if __name__ == "__main__":
    model = yoloResnet(S=7, B=2, C=8)
    summary(model, (16, 3, 448, 448))
    
    
    
    
#### TRY :
if False:
        ### Fully connected part
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-2])
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512 * self.S * self.S, 4096), # 4096 -> 496 modifié le 16/11 
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(0.5), # dropout ajouté le 16/11 
            torch.nn.Linear(4096, self.S * self.S * (self.C + self.B*5)),
            torch.nn.Sigmoid()  # normalized to 0~1 ajouté le 16/11 
        )
