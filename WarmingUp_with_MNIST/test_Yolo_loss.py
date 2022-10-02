import unittest
import torch

from Yolo_loss import YoloLoss

class TestYololoss(unittest.TestCase):
    """
    TODO
    """
    def __init__(self, TestYololoss) -> None:
        super().__init__(TestYololoss)
        self.size = 75
        self.S = 6
        self.B = 1
        self.C = 10
        self.channel_img = 1
        self.BATCH_SIZE = 32

    def test_Yolo_loss(self):
        criterion = YoloLoss(lambd_coord=5, lambd_noobj=0.5, S=self.S, device=torch.device('cpu'))
        box_pred = torch.rand(self.BATCH_SIZE, self.S, self.S, self.B+4)
        box_true = torch.rand(self.BATCH_SIZE, self.S, self.S, self.B+4)
        label_pred = torch.rand(self.BATCH_SIZE, self.S, self.S, self.C)
        label_true = torch.rand(self.BATCH_SIZE, self.C)

        losses, loss = criterion(box_pred, box_true, label_pred, label_true)

        self.assertIs(type(losses), dict)
        self.assertEqual(len(losses), 5)
        self.assertIs(type(loss), torch.Tensor)
        for value in losses.values():
            self.assertIs(type(value), torch.Tensor)



if __name__ == "__main__":
    unittest.main()
        