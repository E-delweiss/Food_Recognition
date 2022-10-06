import glob
import numpy as np
import random as rd
import os

import PIL
import torch
import torch.nn.functional as F
import torchvision

class MealtraysDataset(torch.utils.data.Dataset):
    def __init__(self, root:str, split:str="train", download:bool=False, S=7, B=2, C=8):
        self.S = S
        self.B = B
        self.C = C

        self.SIZE = 448
        self.CELL_SIZE = 1/self.S

        self.root = root
        label_names = {0:'Assiette', 1:'Entree', 2:'Pain', 3:'Boisson', 4:'Yaourt', 5:'Dessert', 6:'Fruit', 7:'Fromage'}
        self.data_txt = glob.glob(root + '/*.txt')

        self.build_annotations(self.data_txt)

        self.FLIP_H = False
        self.FLIP_V = False


    def encode(self, img_path):
        img_name = img_path[img_path.rfind('/')+1 : img_path.rfind('.jpg')]

        # print("\n")
        # print("DEBUG : ", img_name)
        # print("DEBUG : ", self.annotations.get(img_name))
        # print("\n")

        box_target = torch.zeros(self.S, self.S, self.C + 4+1)
        for box in self.annotations.get(img_name):
            ### Relative box infos
            label, xcr_img, ycr_img, wr_img, hr_img = box

            ### Handle flip augmentation
            if self.FLIP_H:
                xcr_img = 1-xcr_img
            if self.FLIP_V:
                ycr_img = 1-ycr_img

            ### Object grid location
            i = np.ceil(xcr_img / self.CELL_SIZE) - 1.0
            j = np.ceil(ycr_img / self.CELL_SIZE) - 1.0
            i, j = int(i), int(j)

            ### x & y of the cell left-top corner
            x0 = i * self.CELL_SIZE
            y0 = j * self.CELL_SIZE
            
            ### x & y of the box on the cell, normalized from 0.0 to 1.0.
            xcr_cell = (xcr_img - x0) / self.CELL_SIZE
            ycr_cell = (ycr_img - y0) / self.CELL_SIZE

            ### Label one-hot encoding
            one_hot_label = F.one_hot(torch.as_tensor(label, dtype=torch.int64), self.C)

            ### 4 coords + 1 conf + 10 classes
            box_target[j, i, :4+1] = torch.Tensor([xcr_cell, ycr_cell, wr_img, hr_img, 1.])
            box_target[j, i, 4+1:] = one_hot_label

        return box_target



    def build_annotations(self, data_txt):
        """
        Example : for key = '20220308112654_000042_059_0000000001_B______xx_C (1636)' 
        Output = [
            [0, 0.414109, 0.291401, 0.412906, 0.565905],
            [5, 0.233531, 0.713405, 0.1885, 0.362759],
            [1, 0.510547, 0.746034, 0.319969, 0.411121],
            [6, 0.742875, 0.764784, 0.158688, 0.243017],
            [3, 0.673172, 0.466121, 0.133281, 0.230948]
            ]
        """
        self.annotations = {}
        for file in data_txt:
            with open(file, 'r') as f:
                obj_list = []
                line_list = f.readline()
                while len(line_list) != 0:
                    line_list = [float(k) for k in line_list.split()]
                    label = int(line_list[0])
                    line_list[0] = label
                    obj_list.append(line_list)
                    line_list = f.readline()
            
            if obj_list:
                start = file.rfind('/')+1
                end = file.rfind('.txt')
                key = file[start : end]
                self.annotations[key] = obj_list


    def convert_to_PIL(self, img_path):
        new_size = (self.SIZE, self.SIZE)
        img = PIL.Image.open(img_path).convert('RGB').resize(new_size, PIL.Image.Resampling.BICUBIC)
        return img


    def flipH(self, img_PIL):
        if rd.random() < 0.5:
            return img_PIL
        img_PIL = torchvision.transforms.RandomHorizontalFlip(p=1.)(img_PIL)
        self.FLIP_H = True
        return img_PIL

    def flipV(self, img_PIL):
        if rd.random() < 0.5:
            return img_PIL
        img_PIL = torchvision.transforms.RandomVerticalFlip(p=1.)(img_PIL)
        self.FLIP_V = True
        return img_PIL

    def augmentation(self, img_PIL):
        img_PIL = self.flipH(img_PIL)
        img_PIL = self.flipV(img_PIL)
        return img_PIL

    def __len__(self):
        return len(self.data_txt)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = self.data_txt[idx][:-4] + ".jpg"
        # print("DEBUG : ", img_path)
        img_PIL = self.convert_to_PIL(img_path)
        img_PIL = self.augmentation(img_PIL)
        img = torchvision.transforms.ToTensor()(img_PIL)
        box_target = self.encode(img_path)

        return img, box_target



if __name__ == '__main__':
    print(os.getcwd())
    dataset = MealtraysDataset(root="plato_dataset/obj_train_data")
    print(dataset[4][1])

