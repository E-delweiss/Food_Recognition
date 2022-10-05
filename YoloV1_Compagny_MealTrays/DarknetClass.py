import torch
import torch.nn as nn
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

class YoloV1(nn.Module):
    def __init__(self, in_channels=1, **kwargs):
        super(YoloV1, self).__init__()
        self.in_channels = in_channels
        self.darknet_params = [
            (7, 64, 2, 3), 
            'M',  # 1
            (3, 192, 1, 1), 
            'M',  # 2
            (1, 128, 1, 0),
            (3, 256, 1, 1), 
            (1, 256, 1, 0),
            (3, 512, 1, 1), 
            'M',  # 3
            [(1, 256, 1, 0), (3, 512, 1, 1), 4], 
            (1, 512, 1, 0), 
            (3, 1024, 1, 1), 
            'M',  # 4
            [(1, 512, 1, 0), (3, 1024, 1, 1), 2], 
            (3, 1024, 1, 1), 
            (3, 1024, 2, 1),
            [(3, 1024, 1, 1), 2]
            ]
        self.darknet = self._create_darknet()
        self.fcs = self._create_fcs(**kwargs)
    

    def _size_output(self, sizeHW, kernel, stride, padding=0):
        output_size = (sizeHW + 2 * padding - (kernel-1)-1)/stride
        output_size = int(output_size + 1)
        return output_size
        
    def _create_darknet(self):
        k=0
        mp=0
        sizeHW = 448
        prev_channel = self.in_channels
        darknet = torch.nn.Sequential()
        for params in self.darknet_params:
            if type(params) is tuple:
                kernel, channel, stride, padding = params
                darknet.add_module(f"CNNBlock_{k}",CNNBlock(in_channels=prev_channel, out_channels=channel, kernel_size=kernel, stride=stride, padding=padding))
                prev_channel = channel
                k+=1
                sizeHW = self._size_output(sizeHW, kernel, stride, padding=padding)
            
            elif type(params) is list:
                for it in range(params[-1]):
                    for sub_params in params[:-1]:
                        kernel, channel, stride, padding = sub_params
                        darknet.add_module(f"CNNBlock_{k}",CNNBlock(in_channels=prev_channel, out_channels=channel, kernel_size=kernel, stride=stride, padding=padding))
                        prev_channel = channel
                        k+=1
                        sizeHW = self._size_output(sizeHW, kernel, stride, padding=padding)
            
            elif params == 'M':
                darknet.add_module(f"MaxPool_{k}", torch.nn.MaxPool2d(kernel_size=2, stride=2))
                mp += 1
                sizeHW = sizeHW / 2

        return darknet


    def _create_fcs(self, split_size, num_boxes, num_classes): #S, B, C
        S, B, C = split_size, num_boxes, num_classes
        output = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(1024*S*S, 4096),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(4096, S*S*(C+B*5))
        )
        return output

    def forward(self, input):
        x = self.darknet(input)
        x = self.fcs(x)
        x = x.view(x.size(0), 7, 7, 1*5+10)
        pc = x[:,:,:,0]
        bbx_coord = x[:,:,:,1:5]
        classifier = x[:,:,:,5:]
        return pc, bbx_coord, classifier



darknet = YoloV1(in_channels=1, split_size=7, num_boxes=1, num_classes=10)
summary(darknet, (2, 1, 448, 448))