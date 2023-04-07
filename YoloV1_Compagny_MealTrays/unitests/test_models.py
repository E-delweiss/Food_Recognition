import unittest
import os, sys
from pathlib import Path

import torch
from torchinfo import summary

current_folder = os.path.dirname(locals().get("__file__"))
parent_folder = Path(current_folder).parent
sys.path.append(str(parent_folder))

from yoloResnet import yoloResnet
from darknet import darknet


class TestModel(unittest.TestCase):
    def __init__(self, TestModel) -> None:
        super().__init__(TestModel)
        self.size = 224
        self.channel_img = 3
        self.S = 7
        self.B = 2
        self.C = 8

    def test_yoloResnet(self):
        BATCH_SIZE = 32
        model = yoloResnet(load_yoloweights=False, S=self.S, C=self.C, B=self.B)
        img_test = torch.rand(BATCH_SIZE, self.channel_img, self.size, self.size)
        output = model(img_test)
        
        # self.assertEqual(output.shape, torch.Size([BATCH_SIZE, self.S, self.S, self.B*(4+1)+self.C]))
        summary(model, input_size = img_test.shape)

    def test_darknet(self):
        BATCH_SIZE = 16
        model = darknet(in_channels=self.channel_img, S=self.S, C=self.C, B=self.B)
        img_test = torch.rand(BATCH_SIZE, self.channel_img, self.size, self.size)
        output = model(img_test)
        
        self.assertEqual(output.shape, torch.Size([BATCH_SIZE, self.S, self.S, self.B*(4+1)+self.C]))
        # summary(model, input_size = img_test.shape)

if __name__ == "__main__":
    unittest.main()
