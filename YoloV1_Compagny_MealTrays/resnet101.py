from configparser import ConfigParser

import torchvision
import torch

from torchinfo import summary

class ResNet(torch.nn.Module):
    def __init__(self, in_channels, S, C, B):
        super(ResNet, self).__init__()
        self.S = S
        self.C = C
        self.B = B

        resnet101 = torchvision.models.resnet50(weights='DEFAULT')
        count=0
        for param in resnet101.parameters():
            param.requires_grad = False
            # if count == 9:
            #     break
            count+=1

        self.seq1 = torch.nn.Sequential(*list(resnet101.children())[:-4])
        self.seq2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(512, 1024, 3, 1, 1),
            torch.nn.Conv2d(1024, 512, 1, 1, 0),
            torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(512, 1024, 3, 1, 1),
            torch.nn.Conv2d(1024, 512, 1, 1, 0),
            torch.nn.MaxPool2d(2,2)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512 * self.S * self.S, 4096),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(4096, self.S * self.S * (self.C + self.B*5))
        )
    
    def forward(self, input):
        x = self.seq1(input)
        x = self.seq2(x)
        x = self.fc(x)
        x = x.view(x.size(0), self.S, self.S, self.B * 5 + self.C)
        return x


if __name__ == "__main__":
    # darknet = resnet101(in_channels=3, S=7, B=2, C=8)
    ResNet = ResNet(3, 7, 8, 2)
    summary(ResNet, (64, 3, 448, 448))

def resnet(pretrained=False, **kwargs) -> ResNet:
    config = ConfigParser()
    config.read("config.ini")
    model_weights = config.get("WEIGHTS", "PT_FILE")

    model = ResNet(**kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model_weights))
    
    return model
    

if __name__ == "__main__":
    model = ResNet(in_channels=3, S=7, B=2, C=8)
    summary(model, (64, 3, 448, 448))