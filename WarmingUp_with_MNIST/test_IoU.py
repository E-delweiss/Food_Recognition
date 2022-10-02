import unittest
import torch
import numpy as np

from IoU import relative2absolute_true, relative2absolute_pred, intersection_over_union

class TestYololoss(unittest.TestCase):
    """
    TODO
    """
    def __init__(self, TestYololoss) -> None:
        super().__init__(TestYololoss)
        size = 75
        S = 6
        B = 1
        C = 10
        channel_img = 1
        BATCH_SIZE = 1
        N = range(BATCH_SIZE)
        self.i, self.j = 2, 3

        self.box_true = torch.zeros(BATCH_SIZE, S, S, B+4)
        self.box_pred = torch.zeros(BATCH_SIZE, S, S, B+4)

        box_value = torch.Tensor([0.52, 0.4, 0.3733, 0.3733, 1])
        
        self.box_true[N,self.i,self.j] = box_value
        self.box_pred[N,self.i,self.j] = box_value

        self.box1 = torch.zeros(BATCH_SIZE, 5)
        self.box2 = torch.zeros(BATCH_SIZE, 5)
        self.box1[:] = torch.Tensor([30,16,57,43,1])
        self.box2[:] = torch.Tensor([30,16,57,43,0.88])


    def test_relative2absolute_true(self):
        box_true_abs = relative2absolute_true(self.box_true)[0]

        self.assertIs(type(box_true_abs), torch.Tensor)
        self.assertEqual(len(box_true_abs), 4)
        self.assertEqual(box_true_abs[0], 30)
        self.assertEqual(box_true_abs[1], 16)
        self.assertEqual(box_true_abs[2], 57)
        self.assertEqual(box_true_abs[3], 43)

    def test_relative2absolute_pred(self):
        box_pred_abs = relative2absolute_pred(self.box_pred, self.i, self.j)[0]

        self.assertIs(type(box_pred_abs), torch.Tensor)
        self.assertEqual(len(box_pred_abs), 4)
        self.assertEqual(box_pred_abs[0], 30)
        self.assertEqual(box_pred_abs[1], 16)
        self.assertEqual(box_pred_abs[2], 57)
        self.assertEqual(box_pred_abs[3], 43)

    
    def test_intersection_over_union(self):
        iou = intersection_over_union(self.box1, self.box2)
        self.assertIs(type(iou), torch.Tensor)
        self.assertEqual(len(iou.shape), 1)
        self.assertEqual(iou[0], 1)



if __name__ == "__main__":
    unittest.main()
        