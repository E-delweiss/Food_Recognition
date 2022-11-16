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
    def __init__(self, in_channels, S, C, B, resnet_pretrained=False):
        super(YoloResNet, self).__init__()
        self.S = S
        self.C = C
        self.B = B

        ### Load ResNet model
        resnet_weights = None
        if resnet_pretrained:
            resnet_weights = 'ResNet34_Weights.DEFAULT'
        resnet = torchvision.models.resnet34(weights=resnet_weights)
        
        ### Freeze ResNet weights
        for param in resnet.parameters():
            param.requires_grad = False

        ### Backbone part
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-2])

        ### Head part
        self.head = torch.nn.Sequential()
        # self.head.add_module("CNNBlock_0",CNNBlock(
        #             in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0))
        self.head.add_module("CNNBlock_1",CNNBlock(
                    in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1))
        self.head.add_module("CNNBlock_2",CNNBlock(
                    in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0))
        self.head.add_module("CNNBlock_3",CNNBlock(
                    in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0))
        
        ### Fully connected part
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(1024 * self.S * self.S, 496), # 4096 -> 496 modifié le 16/11 
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(0.5), # dropout ajouté le 16/11 
            torch.nn.Linear(496, self.S * self.S * (self.C + self.B*5)),
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
    model = yoloResnet(in_channels=3, S=7, B=2, C=8)
    summary(model, (16, 3, 448, 448))