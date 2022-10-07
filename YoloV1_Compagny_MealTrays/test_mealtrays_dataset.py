import unittest
import numpy as np
from mealtrays_dataset import MealtraysDataset

class MealtraysDataset(unittest.TestCase):
    def __init__(self, MealtraysDataset) -> None:
        super().__init__(MealtraysDataset)
        self.size = 448
        self.S = 7
        self.B = 1
        self.C = 8

    def test_my_mnist_dataset(self):
        """
        TODO
        """
        output = MealtraysDataset(root="data", split="train", download=True)
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
        