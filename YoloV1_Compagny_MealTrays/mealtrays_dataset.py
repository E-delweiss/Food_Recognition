import glob
import numpy as np
import random as rd
import os

import PIL
import torch
import torch.nn.functional as F
import torchvision
from icecream import ic
import albumentations as A

class MealtraysDataset(torch.utils.data.Dataset):
    def __init__(self, root:str, split:str="train", isNormalize:bool=False, isAugment:bool=False, S=7, C=8):
        """
        label_names = {0:'Assiette', 1:'Entree', 2:'Pain', 3:'Boisson', 4:'Yaourt', 5:'Dessert', 6:'Fruit', 7:'Fromage'}
        """
        ### Yolo output params S:grid ; B:boxes ; C:classes
        self.S = S
        self.C = C

        ### Sizes
        self.SIZE_TEMP = 330
        self.SIZE = 224 #448
        self.CELL_SIZE = 1/self.S
        self.mean = torch.tensor([0.4167, 0.4045, 0.3833])
        self.std = torch.tensor([0.3480, 0.3439, 0.3379])

        ### Get data
        self.root = root

        ### Get train/validation
        if split == 'train':
            data_txt = glob.glob(root + '/train/*.txt')
            assert len(data_txt) != 0, "\nError : the path may be wrong."
        else :
            data_txt = glob.glob(root + '/val/*.txt')
            assert len(data_txt) != 0, "\nError : the path may be wrong."

        ### Build annotations for an image as a dict and 
        ### get only data_txt that has been labelised
        self.annotations, self.data_txt_labelised = self._build_annotations(data_txt)

        ### Data augmentation
        self.isAugment = isAugment

        ### Data normalization
        self.isNormalize = isNormalize

        ### Only to generate mean/std
        # self._mean_std_fct()

    def _build_annotations(self, data_txt):
        """
        Example : for key = '20220308112654_000042_059_0000000001_B______xx_C (1636)' 
        Output = [
            [0.414109, 0.291401, 0.412906, 0.565905, 0],
            [0.233531, 0.713405, 0.1885, 0.362759, 5],
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
                    label = int(line_list.pop(0))
                    line_list.append(label)
                    # Append to obj_list which contains all the params for 
                    # one object in the current image
                    obj_list.append(line_list)
                    line_list = f.readline()
            
            ### Construct the dict
            if obj_list:
                data_txt_labelised.append(file)
                start, end = file.rfind('/')+1, file.rfind('.txt')
                key = file[start : end] # -> key is imgname
                annotations[key] = obj_list
        
        return annotations, data_txt_labelised

    def _convert_to_PIL(self, img_path):
        new_size = (self.SIZE, self.SIZE)
        img = PIL.Image.open(img_path).convert('RGB').resize(new_size, PIL.Image.Resampling.BICUBIC)
        return img

    def _mean_std_fct(self):
        """
        Only to generate mean/std
        """
        data_PIL = [PIL.Image.open(img_path.replace(".txt", ".jpg")).convert('RGB') for img_path in self.data_txt_labelised]
        data_tensor = [torchvision.transforms.ToTensor()(img_PIL) for img_PIL in data_PIL]

        channels_sum, channels_squared_sum = 0, 0
        for img in data_tensor:
            channels_sum += torch.mean(img, dim=[1,2])
            channels_squared_sum += torch.mean(img**2, dim=[1,2])
        
        mean = channels_sum/len(data_tensor)
        std = torch.sqrt((channels_squared_sum/len(data_tensor) - mean**2))
        
        print("MEAN AND STD OF LABELLISED DATASET : ")
        print(mean, std)

        return mean, std

    def _unNormalize(self, img_tensor):
        inv_normalize = torchvision.transforms.Normalize(
            mean = -self.mean/self.std,
            std = 1/self.std
            )
        img_tensor = inv_normalize(img_tensor)
        img_PIL = torchvision.transforms.ToPILImage()(img_tensor)
        return img_PIL

    def _process(self, img_path):
        """
        TODO
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
        annotations = self.annotations.get(img_name)
        img_PIL = self._convert_to_PIL(img_path)
        img_tensor = torchvision.transforms.ToTensor()(img_PIL)

        if self.isNormalize:
            img_tensor = torchvision.transforms.Normalize(
                mean=self.mean, std=self.std
                )(img_tensor)

        if self.isAugment:
            ### ALBUMENTATION
            albumentation = A.Compose([
                A.RandomResizedCrop(width=self.SIZE, height=self.SIZE, scale=(0.5, 1), p=1),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=[-0.2,0.1], p=0.5),
                ], bbox_params=A.BboxParams(format='yolo', min_visibility = 0.4)
                )

            albu_dict = albumentation(image=np.array(img_tensor.permute(1,2,0)), bboxes=annotations)
            img_tensor = torch.Tensor(albu_dict['image']).permute(2,0,1)
            annotations = albu_dict['bboxes']


        target = torch.zeros(self.S, self.S, self.C + 4+1)
        for target_infos in annotations:
            ### Relative box infos
            xcr_img, ycr_img, wr_img, hr_img, label = target_infos

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
        
        return img_tensor, target

    def __len__(self):
        return len(self.data_txt_labelised)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data_txt_labelised[idx].replace(".txt", ".jpg")
        img, target = self._process(img_path)

        return img, target



def get_training_dataset(BATCH_SIZE=16, **kwargs):
    """
    Loads and maps the training split of the dataset using the custom dataset class. 
    """
    dataset = MealtraysDataset(root="../data/mealtray_dataset", **kwargs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

def get_validation_dataset(BATCH_SIZE=None, **kwargs):
    """
    Loads and maps the validation split of the datasetusing the custom dataset class. 
    """
    dataset = MealtraysDataset(root="../data/mealtray_dataset", **kwargs)
    if BATCH_SIZE is None:
        BATCH_SIZE = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return dataloader


if __name__ == "__main__":
    # dataset = MealtraysDataset(root="../data/mealtray_dataset", isAugment=False, isNormalize=False)
    # dataset._mean_std_fct()
    dataloader = get_training_dataset(BATCH_SIZE=8, isAugment=True, isNormalize=True)
    a, b = next(iter(dataloader))
    ic(a.shape, b.shape)