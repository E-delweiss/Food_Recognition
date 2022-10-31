import unittest
import os, sys
from pathlib import Path

import torch
from torchinfo import summary

current_folder = os.path.dirname(locals().get("__file__"))
parent_folder = Path(current_folder).parent
sys.path.append(str(parent_folder))

import darknet
import darknet_like
import darknet_like2


class TestDarknet(unittest.TestCase):
    def __init__(self, TestDarknet) -> None:
        super().__init__(TestDarknet)
        self.size = 448
        self.S = 7
        self.B = 2
        self.C = 8
        self.channel_img = 3

    def test_darknet(self):
        BATCH_SIZE = 64
        model = darknet.YoloV1(in_channels=self.channel_img, S=self.S, C=self.C, B=self.B)
        img_test = torch.rand(BATCH_SIZE, self.channel_img, self.size, self.size)
        output = model(img_test)
        
        self.assertEqual(output.shape, torch.Size([BATCH_SIZE, self.S, self.S, self.B*(4+1)+self.C]))
        #summary(model, input_size = img_test.shape)


    def test_darknet_like(self):
        BATCH_SIZE = 64
        model = darknet_like.YoloV1(sizeHW=self.size, S=self.S, C=self.C, B=self.B)
        img_test = torch.rand(BATCH_SIZE, self.channel_img, self.size, self.size)
        output = model(img_test)
        
        self.assertEqual(output.shape, torch.Size([BATCH_SIZE, self.S, self.S, self.B*(4+1)+self.C]))
        # summary(model, input_size = img_test.shape)

    def test_darknet_like2(self):
        BATCH_SIZE = 64
        model = darknet_like2.YoloV1(in_channels=3, S=self.S, C=self.C, B=self.B)
        img_test = torch.rand(BATCH_SIZE, self.channel_img, self.size, self.size)
        output = model(img_test)
        
        self.assertEqual(output.shape, torch.Size([BATCH_SIZE, self.S, self.S, self.B*(4+1)+self.C]))
        # summary(model, input_size = img_test.shape)

if __name__ == "__main__":
    unittest.main()
