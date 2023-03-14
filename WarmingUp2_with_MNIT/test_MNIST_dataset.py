import unittest
import numpy as np
from MNIST_dataset import MNISTDataset

class TestMNISTDataset(unittest.TestCase):
    def __init__(self, TestMNISTDataset) -> None:
        super().__init__(TestMNISTDataset)
        self.size = 75
        self.S = 6
        self.B = 1
        self.C = 10

    def test_my_mnist_dataset(self):
        output = MNISTDataset(root="data", split="train", download=True)
        idx = np.random.randint(len(output))
        output = output[idx]
        self.assertIs(type(output), tuple)
        self.assertEqual(len(output), 3)

        self.assertEqual(len(output[0].shape), 3)
        self.assertEqual(output[0].shape[1], self.size)
        self.assertEqual(output[0].shape[1], output[0].shape[2])
        
        self.assertEqual(len(output[1].shape), 3)
        self.assertEqual(output[1].shape[0], self.S)
        self.assertEqual(output[1].shape[0], output[1].shape[1])
        self.assertEqual(output[1].shape[2], self.B + 4)

        self.assertEqual(len(output[2].shape), 1)
        self.assertEqual(output[2].shape[0], self.C)

if __name__ == "__main__":
    unittest.main()
        