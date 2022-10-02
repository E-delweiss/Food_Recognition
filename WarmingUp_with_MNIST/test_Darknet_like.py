import unittest
import torch

from Darknet_like import YoloMNIST

class TestDarknetlike(unittest.TestCase):
    """
    TODO
    """
    def __init__(self, TestDarknetlike) -> None:
        super().__init__(TestDarknetlike)
        self.size = 75
        self.S = 6
        self.B = 1
        self.C = 10
        self.channel_img = 1

    def test_YoloMNIST(self):
        BATCH_SIZE = 64
        model = YoloMNIST(sizeHW=self.size, S=self.S, C=self.C, B=self.B)
        img_test = torch.rand(BATCH_SIZE, self.channel_img, self.size, self.size)
        box_pred, label_pred = model(img_test)
        
        self.assertEqual(box_pred.shape, torch.Size([BATCH_SIZE, self.S, self.S, self.B+4]))
        self.assertEqual(label_pred.shape, torch.Size([BATCH_SIZE, self.S, self.S, self.C]))


if __name__ == "__main__":
    unittest.main()
        