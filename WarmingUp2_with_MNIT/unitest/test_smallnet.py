import unittest
import torch

from smallnet import NetMNIST

current_folder = os.path.dirname(locals().get("__file__"))
parent_folder = Path(current_folder).parent
sys.path.append(str(parent_folder))

class TestSmallNet(unittest.TestCase):
    def __init__(self, TestDarknetlike) -> None:
        super().__init__(TestDarknetlike)
        self.size = 75
        self.S = 6
        self.B = 1
        self.C = 10
        self.channel_img = 1

    def test_smallnet(self):
        BATCH_SIZE = 64
        model = NetMNIST(sizeHW=self.size, S=self.S, C=self.C, B=self.B)
        img_test = torch.rand(BATCH_SIZE, self.channel_img, self.size, self.size)
        box_pred, label_pred = model(img_test)
        
        self.assertEqual(box_pred.shape, torch.Size([BATCH_SIZE, self.S, self.S, self.B+4]))
        self.assertEqual(label_pred.shape, torch.Size([BATCH_SIZE, self.S, self.S, self.C]))


if __name__ == "__main__":
    unittest.main()
        
