import unittest
import torch

from metrics import class_acc, MSE, MSE_confidenceScore

class TestYololoss(unittest.TestCase):
    def __init__(self, TestYololoss) -> None:
        super().__init__(TestYololoss)
        S = 7
        B = 2
        C = 8
        BATCH_SIZE = 32

        self.target = torch.zeros(BATCH_SIZE, S, S, 5+C)
        N = range(BATCH_SIZE)
        i = torch.randint(S, (1,BATCH_SIZE))
        j = torch.randint(S, (1,BATCH_SIZE))
        box_value = torch.rand((BATCH_SIZE, 5))
        self.target[N,i,j,:5] = box_value

        label_true = torch.zeros(BATCH_SIZE, C)
        idx = torch.randint(C, (1, BATCH_SIZE))
        label_true[N, idx] = 1
        self.target[N,i,j,5:] = label_true
        
        self.prediction = torch.zeros(BATCH_SIZE, S, S, 5*B+C)
        self.prediction[N,i,j,:5] = box_value
        self.prediction[N,i,j,5:10] = box_value
        self.prediction[:,:,:,10:] = torch.rand(8)
        self.prediction[N,i,j,10:] = label_true

        self.N, self.i, self.j = N, i, j

    def test_class_acc(self):
        prediction = self.prediction.clone()
        prediction[self.N, self.i, self.j, 10:] *= 0.999
        acc = class_acc(self.target, self.prediction)
        self.assertIs(type(acc), float)
        self.assertGreaterEqual(acc, 0.)
        self.assertLessEqual(acc,1.)
        self.assertAlmostEqual(acc, 1)

    def test_MSE(self):
        variations = [[0.9995, 0.5], [0.5, 0.9995]]
        for value in variations:
            prediction = self.prediction.clone()
            prediction[self.N, self.i, self.j, :5] *= value[0]
            prediction[self.N, self.i, self.j, 5:10] *= value[1]
            mse = MSE(self.target, prediction)
            self.assertIs(type(mse), float)
            self.assertGreaterEqual(mse, 0.)
            self.assertAlmostEqual(mse, 0, places=5)

    def test_MSE_confidenceScore(self):
        variations = [[0.9995, 0.5], [0.5, 0.9995]]
        for value in variations:
            prediction = self.prediction.clone()
            prediction[self.N, self.i, self.j, 4] *= value[0]
            prediction[self.N, self.i, self.j, 9] *= value[1]

            mse = MSE_confidenceScore(self.target, self.prediction)
            self.assertIs(type(mse), float)
            self.assertGreaterEqual(mse, 0.)
            self.assertAlmostEqual(mse, 0, places=5)


if __name__ == "__main__":
    unittest.main()
        
