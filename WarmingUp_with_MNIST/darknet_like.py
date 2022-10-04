import torch
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


class YoloMNIST(torch.nn.Module):
    def __init__(self, sizeHW, S, C, B):
        super(YoloMNIST, self).__init__()
        self.S, self.C, self.B = S, C, B
        self.sizeHW = sizeHW
        self.cell_size = self.sizeHW / self.S

        self.seq = torch.nn.Sequential()        
        self.seq.add_module(f"conv_1", CNNBlock(1, 32, stride=2, kernel_size=7, padding=2))
        self.seq.add_module(f"maxpool_1", torch.nn.MaxPool2d(2))
        self.seq.add_module(f"conv_2", CNNBlock(32, 128, stride=1, kernel_size=3, padding=0))
        self.seq.add_module(f"maxpool_2", torch.nn.MaxPool2d(2))
        self.seq.add_module(f"conv_3", CNNBlock(128, 64, stride=1, kernel_size=1, padding=0))
        self.seq.add_module(f"conv_4", CNNBlock(64, 128, stride=1, kernel_size=3, padding=0))
        self.seq.add_module(f"conv_5", CNNBlock(128, 128, stride=1, kernel_size=3, padding=1))
        
        self.fcs = self._create_fcs()

    def _size_output(self, sizeHW:int, kernel:int, stride:int, padding:int=0, isMaxPool:bool=False)->int:
        """
        Output size (width/height) of convolutional or maxpool layers.

        Args:
            sizeHW : int
                Image size (we suppose this is a square image)
            kernel : int
                Size of a square kernel
            stride : int
                Stride of convolution layer
            padding : int
                Padding of convolution layer
            isMaxPool : Bool, default is False.
                Specify if it is a Maxpool layer (True) or not (False). 

        Return:
            output_size : int
                Image output size after a convolutional or MaxPool layer.
        """ 
        if isMaxPool == True:
            output_size = int(sizeHW/2)
            print(output_size)
            return output_size
        if padding == 'same':
            output_size = sizeHW
            print(output_size)
            return output_size
        else:
            output_size = (sizeHW + 2 * padding - (kernel-1)-1)/stride
            output_size = int(output_size + 1)
            print(output_size)
            return output_size

    def _create_fcs(self):
        output = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * self.S * self.S, 4096),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(4096, self.S * self.S * (self.C + self.B*5))
        )
        return output
    

    def forward(self, input:torch.Tensor)->tuple:
        """
        Forward pass.

        Args:
            input : torch.Tensor of shape (N, C, H, W)
                Batch of images.

        Return:
            box_coord : torch.Tensor of shape (N, 6, 6, 5)
                Contains xc_rcell, yc_rcell, rw, rh and the confidence number c
                over 6x6 grid cells.
            classifier : torch.Tensor of shape (N, 6, 6, 10)
                Contains the one-hot encoding of each digit number over
                6x6 grid cells.
        """     
        x = self.seq(input)
        x = self.fcs(x)
        x = x.view(x.size(0), self.S, self.S, self.B * 5 + self.C)
        box_coord = x[:,:,:,0:5]
        classifier = x[:,:,:,5:]
        return box_coord, classifier


if __name__ == "__main__":
    model = YoloMNIST(sizeHW=75, S=6, C=10, B=1)

    BATCH_SIZE = 64
    img_test = torch.rand(BATCH_SIZE, 1, 75, 75)
    summary(model, input_size = img_test.shape)
