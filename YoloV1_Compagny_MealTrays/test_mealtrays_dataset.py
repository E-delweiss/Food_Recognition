import unittest
import numpy as np
from mealtrays_dataset import MealtraysDataset

class TestMealtraysDataset(unittest.TestCase):
    def __init__(self, TestMealtraysDataset) -> None:
        super().__init__(TestMealtraysDataset)
        self.SIZE = 448
        self.S = 7
        self.B = 1
        self.C = 8

    def test_my_mealtrays_dataset(self):
        output = MealtraysDataset(root="YoloV1_Compagny_MealTrays/mealtrays_dataset", split="train")        
        idx = np.random.randint(len(output))
        output = output[idx]

        ### Test on output type/size
        self.assertIs(type(output), tuple)
        self.assertEqual(len(output), 2)

        ### Test on output image shape
        self.assertEqual(len(output[0].shape), 3)
        self.assertEqual(output[0].shape[1], output[0].shape[2])
        
        ### Test on output target shape
        self.assertEqual(len(output[1].shape), 3)
        self.assertEqual(output[1].shape[0], self.S)
        self.assertEqual(output[1].shape[0], output[1].shape[1])
        self.assertEqual(output[1].shape[2], self.B*(4+1) + self.C)

if __name__ == "__main__":
    unittest.main()
        