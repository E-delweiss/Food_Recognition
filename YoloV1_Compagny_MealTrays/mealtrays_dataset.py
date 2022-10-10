import glob
import numpy as np
import random as rd
import os

import PIL
import torch
import torch.nn.functional as F
import torchvision

class MealtraysDataset(torch.utils.data.Dataset):
    def __init__(self, root:str, split:str="train", isNormalize:bool=True, isAugment:bool=True, S=7, C=8):
        """
        label_names = {0:'Assiette', 1:'Entree', 2:'Pain', 3:'Boisson', 4:'Yaourt', 5:'Dessert', 6:'Fruit', 7:'Fromage'}
        """
        ### Yolo output params S:grid ; B:boxes ; C:classes
        self.S = S
        self.C = C

        ### Sizes
        self.SIZE = 448
        self.CELL_SIZE = 1/self.S
        
        ### Get data
        self.root = root
        data_txt = glob.glob(root + '/obj_train_data/*.txt')
        assert len(data_txt) != 0, "\nError : the path may be wrong."
        for data in data_txt:
            assert os.path.isfile(data), "\nError : the file {data} does not exist."

        ### Get train/validation
        if split == 'train':
            split_pct = 0.8
            length_data = round(len(data_txt) * split_pct)
            data_txt = data_txt[:length_data]
        else :
            split_pct = 0.2
            length_data = round(len(data_txt) * split_pct)
            data_txt = data_txt[-length_data:]

        ### Build annotations for an image as a dict and 
        ### get only data_txt that has been labelised
        self.annotations, self.data_txt_labelised = self._build_annotations(data_txt)

        ### Data augmentation. TODO : crop
        self.isAugment = isAugment
        self.FLIP_H = False
        self.FLIP_V = False
        self.CROP = False

        ### Data normalization. See utils.py module
        self.isNormalize = isNormalize

    def _build_annotations(self, data_txt):
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
        annotations = {}
        data_txt_labelised = []
        ### Loop on each file .txt
        for file in data_txt:
            with open(file, 'r') as f:
                obj_list = []
                line_list = f.readline()
                # Read line by line until there are no more
                while len(line_list) != 0:
                    # Read params in the line
                    line_list = [float(k) for k in line_list.split()]
                    label = int(line_list[0])
                    line_list[0] = label
                    # Append to obj_list which contains all the params for 
                    # one object in the current image
                    obj_list.append(line_list)
                    line_list = f.readline()
            
            ### Construct the dict
            if obj_list:
                data_txt_labelised.append(file)
                start, end = file.rfind('/')+1, file.rfind('.txt')
                key = file[start : end] #key = img name
                annotations[key] = obj_list
        
        return annotations, data_txt_labelised

    def _convert_to_PIL(self, img_path):
        new_size = (self.SIZE, self.SIZE)
        img = PIL.Image.open(img_path).convert('RGB').resize(new_size, PIL.Image.Resampling.BICUBIC)
        return img

    def _transform(self, img_PIL):
        img_tensor = torchvision.transforms.ToTensor()(img_PIL)
        if self.isNormalize:
            torchvision.transforms.Normalize(
                mean=(0.4111, 0.4001, 0.3787), std=(0.3461, 0.3435, 0.3383)
                )(img_tensor)
        return img_tensor

    def _flipH(self, img_PIL):
        if rd.random() < 0.5:
            self.FLIP_H = False
            return img_PIL
        img_PIL = torchvision.transforms.RandomHorizontalFlip(p=1.)(img_PIL)
        self.FLIP_H = True
        return img_PIL

    def _flipV(self, img_PIL):
        if rd.random() < 0.5:
            self.FLIP_V = False
            return img_PIL
        img_PIL = torchvision.transforms.RandomVerticalFlip(p=1.)(img_PIL)
        self.FLIP_V = True
        return img_PIL

    def _crop(self, img_PIL):
        if rd.random() < 0.5:
            self.CROP = False
            return img_PIL
        crop_size = (400,400)
        crop_infos = torchvision.transforms.RandomCrop.get_params(img_PIL, crop_size)
        img_PIL = torchvision.transforms.functional.crop(img_PIL, *crop_infos)
        self.CROP = True 
        return img_PIL, crop_infos

    def _augmentation(self, img_PIL):
        img_PIL = self._flipH(img_PIL)
        img_PIL = self._flipV(img_PIL)
        img_PIL = self._crop(img_PIL)
        return img_PIL

    def _encode(self, img_path, crop_infos=()):
        """
        Encode box informations (coordinates, size and label) as a 
        (S,S,C+B+4+1) tensor.

        Args:
            img_path (str)
                Absolute path of image.jpg

        Returns:
            target: torch.Tensor of shape (S,S,14)
                Tensor of zeros representing the SxS grid. Zeros everywhere
                except in the (j,i) positions where there is an object.
        """
        ### Retrieve image name (without '.jpg') from path 
        img_name = img_path[img_path.rfind('/')+1 : img_path.rfind('.jpg')]

        target = torch.zeros(self.S, self.S, self.C + 4+1)
        for target_infos in self.annotations.get(img_name):
            ### Relative box infos
            label, xcr_img, ycr_img, wr_img, hr_img = target_infos

            ### Handle flip augmentation
            if self.FLIP_H:
                xcr_img = 1-xcr_img
            if self.FLIP_V:
                ycr_img = 1-ycr_img
            if self.CROP:
                x_crop = crop_infos[1]/self.SIZE
                y_crop = crop_infos[0]/self.SIZE
                wh_crop = crop_infos[2]
                xcr_img = xcr_img - x_crop
                ycr_img = ycr_img - y_crop
                xcr_img = np.clip(xcr_img, 0, wh_crop)

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

            ### 4 coords + 1 conf + 8 classes
            target[j, i, :4+1] = torch.Tensor([xcr_cell, ycr_cell, wr_img, hr_img, 1.])
            target[j, i, 4+1:] = one_hot_label
        return target

    def __len__(self):
        return len(self.data_txt_labelised)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        crop_infos = ()
        img_path = self.data_txt_labelised[idx].replace(".txt", ".jpg")
        
        img_PIL = self._convert_to_PIL(img_path)
        if self.isAugment:
            img_PIL = self._augmentation(img_PIL)
            if self.CROP:
                img_PIL, crop_infos = img_PIL
        
        img = self._transform(img_PIL)
        target = self._encode(img_path, crop_infos)

        return img, target


def get_training_dataset(BATCH_SIZE=16):
    """
    Loads and maps the training split of the dataset using the custom dataset class. 
    """
    dataset = MealtraysDataset(root="YoloV1_Compagny_MealTrays/mealtrays_dataset", split="train", isNormalize=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

def get_validation_dataset(BATCH_SIZE=None):
    """
    Loads and maps the validation split of the datasetusing the custom dataset class. 
    """
    dataset = MealtraysDataset(root="YoloV1_Compagny_MealTrays/mealtrays_dataset", split="test", isNormalize=True, isAugment=False)
    if BATCH_SIZE is None:
        BATCH_SIZE = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return dataloader