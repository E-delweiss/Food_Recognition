import unittest
import os, sys
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision

current_folder = os.path.dirname(locals().get("__file__"))
parent_folder = Path(current_folder).parent
sys.path.append(str(parent_folder))

from mealtrays_dataset import MealtraysDataset

class TestMealtraysDataset(unittest.TestCase):
    def __init__(self, TestMealtraysDataset) -> None:
        super().__init__(TestMealtraysDataset)
        self.SIZE = 448
        self.S = 7
        self.B = 1
        self.C = 8
        self.CELL_SIZE = 1/self.S

        dataset = MealtraysDataset(root="../../mealtray_dataset/dataset",
            split="train", isNormalize=True, isAugment=True)        
        idx = np.random.randint(len(dataset))
        self.output = dataset[idx]

    def test_my_mealtrays_dataset(self):
        ### Test on output type/size
        self.assertIs(type(self.output), tuple)
        self.assertEqual(len(self.output), 2)

        ### Test on output image shape
        self.assertEqual(len(self.output[0].shape), 3)
        self.assertEqual(self.output[0].shape[1], self.output[0].shape[2])
        
        ### Test on output target shape
        self.assertEqual(len(self.output[1].shape), 3)
        self.assertEqual(self.output[1].shape[0], self.S)
        self.assertEqual(self.output[1].shape[0], self.output[1].shape[1])
        self.assertEqual(self.output[1].shape[2], self.B*(4+1) + self.C)

    def test_plot_dataset(self):
        color_dict = {'Plate':'b', 'Starter':'g', 'Bread':'r', 'Drink':'c', 
            'Yogurt':'darkred', 'Dessert':'k', 'Fruit':'m', 'Cheese':'y'}
        label_dict = {0:'Plate', 1:'Starter', 2:'Bread', 3:'Drink', 
            4:'Yogurt', 5:'Dessert', 6:'Fruit', 7:'Cheese'}

        img_idx, target = self.output

        # cailculate mean and std
        mean, std = img_idx.mean([1,2]), img_idx.std([1,2])
        
        # print mean and std
        print("\nMean and Std of normalized image:")
        print("Mean of the image:", mean)
        print("Std of the image:", std)

        inv_normalize = torchvision.transforms.Normalize(
        mean=[-0.4168/0.3475, -0.4055/0.3442, -0.3838/0.3386],
        std=[1/0.3475, 1/0.3442, 1/0.3386]
        )
        img_idx = inv_normalize(img_idx)
        img_idx = torchvision.transforms.ToPILImage()(img_idx)
        
        cells_i, cells_j, _ = target.nonzero().permute(1,0)
        cells_with_obj = torch.stack((cells_i, cells_j), dim=1)
        cells_i, cells_j = torch.unique(cells_with_obj,dim=0).permute(1,0)

        box_infos = target[cells_i, cells_j, :]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img_idx)
        ax.set_xticks = []
        ax.set_yticks = []
        ax.set_axis_off()

        k = 0
        for box in box_infos:
            label = torch.argmax(box[(4+1):], dim=0).item()
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

            color = color_dict.get(label_dict.get(label))
            rect = patches.Rectangle((xmin, ymin), w, h, facecolor='none', edgecolor=color)
            ax.add_patch(rect)
            offset_x = 2
            offset_y = 10
            rect_txt = patches.Rectangle((xmin, ymin), w, 15, facecolor=color, edgecolor=color)
            ax.add_patch(rect_txt)
            ax.text(xmin+offset_x, ymin+offset_y, label_dict.get(label), fontsize=8, color='w', family='monospace', weight='bold')
            
        plt.show()


if __name__ == "__main__":
    unittest.main()
