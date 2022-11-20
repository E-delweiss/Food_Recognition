from configparser import ConfigParser

import torch
import torchvision
from torchinfo import summary

class CNNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.l_relu = torch.nn.LeakyReLU(0.1)
    
    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        return self.l_relu(x)

class YoloResNet(torch.nn.Module):
    def __init__(self, S, C, B, resnet_pretrained=False):
        super(YoloResNet, self).__init__()
        self.S = S
        self.C = C
        self.B = B

        ### Load ResNet model
        resnet_weights = None
        if resnet_pretrained:
            resnet_weights = 'DEFAULT'
        resnet = torchvision.models.resnet152(weights=resnet_weights)
        
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


def yoloResnet(load_yoloweights=False, resnet_pretrained=True, **kwargs) -> YoloResNet:
    assert load_yoloweights != resnet_pretrained, "Can't load both ResNetYolo weights and ResNet50 weights"
    
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