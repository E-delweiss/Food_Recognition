import unittest
import numpy as np

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from mealtrays_dataset import MealtraysDataset

class TestMealtraysDataset(unittest.TestCase):
    def __init__(self, TestMealtraysDataset) -> None:
        super().__init__(TestMealtraysDataset)
        self.SIZE = 448
        self.S = 7
        self.B = 1
        self.C = 8
        self.CELL_SIZE = 1/self.S

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

    def test_plot_dataset(self):
        output = MealtraysDataset(root="YoloV1_Compagny_MealTrays/mealtrays_dataset", split="train", isNormalize=False, isAugment=False)        
        idx = np.random.randint(len(output))
        img_idx, tensor_grid = output[idx]

        color_dict = {'Assiette':'b', 'Entree':'g', 'Pain':'r', 'Boisson':'c', 
        'Yaourt':'w', 'Dessert':'k', 'Fruit':'m', 'Fromage':'y'}
        label_dict = {0:'Assiette', 1:'Entree', 2:'Pain', 3:'Boisson', 
        4:'Yaourt', 5:'Dessert', 6:'Fruit', 7:'Fromage'}

        import torchvision
        # img_idx = img_idx * 255.0
        img_idx = torchvision.transforms.ToPILImage()(img_idx)

        cells_with_obj = tensor_grid.nonzero()[::tensor_grid.shape[-1]]
        cells_i, cells_j, _ = cells_with_obj.permute(1,0)
        box_infos = tensor_grid[cells_i, cells_j, :]

        ax = plt.figure()
        ax.imshow(img_idx)
        ax.set_xticks = []
        ax.set_yticks = []

        k = 0
        for box in box_infos:
            label_name = torch.argmax(box[(4+1):], dim=0)
            ### Get relative positions
            xcr_cell = box[0]
            ycr_cell = box[1]
            wr_img = box[2]
            hr_img = box[3]
            
            ### Create absolute positions     
            xcr_img =  xcr_cell * self.CELL_SIZE + cells_j[k] * self.CELL_SIZE
            ycr_img =  ycr_cell * self.CELL_SIZE + cells_i[k] * self.CELL_SIZE   
            k += 1

            xmin = (xcr_img - wr_img/2) * self.SIZE
            ymin = (ycr_img - hr_img/2) * self.SIZE

            ### Create absolute width and height
            w = wr_img * self.SIZE
            h = hr_img * self.SIZE

            color = color_dict.get(label_name)
            rect = patches.Rectangle((xmin, ymin), w, h, facecolor='none', edgecolor=color)
            ax.add_patch(rect)

            offset_x = 2
            offset_y = 15
            ax.text(xmin+offset_x, ymin+offset_y, label_dict.get(label_name), fontsize=8, color=color, family='monospace', weight='bold')

        plt.show()


if __name__ == "__main__":
    unittest.main()
        