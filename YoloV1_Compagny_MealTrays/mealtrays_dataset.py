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
        self.SIZE_TEMP = 500
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

        ### Data augmentation.
        self.isAugment = isAugment
        self.FLIP_H = False
        self.FLIP_V = False
        self.CROP = False
        self.HARDCROP = False

        ### Data normalization. See utils.py module
        self.isNormalize = isNormalize

        ### Only to generate mean/std
        # self._mean_std_fct()

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
                key = file[start : end] # -> key is imgname
                annotations[key] = obj_list
        
        return annotations, data_txt_labelised

    def _convert_to_PIL(self, img_path):
        new_size = (self.SIZE, self.SIZE)
        if self.isAugment:
            new_size = (self.SIZE_TEMP, self.SIZE_TEMP)

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

    def _transform(self, img_PIL):
        img_tensor = torchvision.transforms.ToTensor()(img_PIL)
        if self.isNormalize:
            img_tensor = torchvision.transforms.Normalize(
                mean=(0.4168, 0.4055, 0.3838), std=(0.3475, 0.3442, 0.3386)
                )(img_tensor)
        return img_tensor

    def _flipH(self, img_PIL):
        self.FLIP_H = True
        if rd.random() < 0.5:
            self.FLIP_H = False
            return img_PIL
        img_PIL = torchvision.transforms.RandomHorizontalFlip(p=1.)(img_PIL)
        return img_PIL

    def _flipV(self, img_PIL):
        self.FLIP_V = True
        if rd.random() < 0.5:
            self.FLIP_V = False
            return img_PIL
        img_PIL = torchvision.transforms.RandomVerticalFlip(p=1.)(img_PIL)
        return img_PIL

    def _crop(self, img_PIL):
        self.CROP = True 
        if rd.random() < 0.5:
            self.CROP = False
            new_size = (self.SIZE, self.SIZE)
            img_PIL = img_PIL.resize(new_size, PIL.Image.Resampling.BICUBIC)
            return img_PIL
        
        crop_size = (self.SIZE, self.SIZE)
        crop_infos = list(torchvision.transforms.RandomCrop.get_params(img_PIL, crop_size))
        img_PIL = torchvision.transforms.functional.crop(img_PIL, *crop_infos)
        return img_PIL, crop_infos

    def _hardcrop(self, img_PIL):
        self.HARDCROP = True
        if rd.random() < 0.00001:
            self.HARDCROP = False
            new_size = (self.SIZE, self.SIZE)
            img_PIL = img_PIL.resize(new_size, PIL.Image.Resampling.BICUBIC)
            return img_PIL

        crop_size = (400, 400)
        crop_infos = list(torchvision.transforms.RandomCrop.get_params(img_PIL, crop_size))
        img_PIL = torchvision.transforms.functional.crop(img_PIL, *crop_infos)
        img_PIL = img_PIL.resize((448,448), PIL.Image.Resampling.BICUBIC)

        # offset = 448 - 400
        # crop_infos[1] += offset
        # crop_infos[0] += offset

        print("\nDEBUG: ", crop_infos)

        return img_PIL, crop_infos

    def _augmentation(self, img_PIL):
        img_PIL = self._flipH(img_PIL)
        img_PIL = self._flipV(img_PIL)
        img_PIL = self._crop(img_PIL)
        # img_PIL = self._hardcrop(img_PIL)
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

            ### Handle random crop (500,500) -> (448,448)
            if self.CROP:
                posXcrop_rimg = crop_infos[1]/self.SIZE
                posYcrop_rimg = crop_infos[0]/self.SIZE
            
                ### Compute absolute coord in 500x500 img
                xc = xcr_img * self.SIZE_TEMP
                yc = ycr_img * self.SIZE_TEMP
            
                ### Compute relative coord to 448x448 img and handle cropping
                xcr_img = xc/self.SIZE - posXcrop_rimg
                ycr_img = yc/self.SIZE - posYcrop_rimg

                ### Restrict the cropping btw 0 & 1
                xcr_img = np.clip(xcr_img, 0, 1)
                ycr_img = np.clip(ycr_img, 0, 1)

            if self.HARDCROP: #TODO
                posXcrop_rimg = crop_infos[1]/self.SIZE
                posYcrop_rimg = crop_infos[0]/self.SIZE
            
                ### Compute absolute coord in 500x500 img
                xc = xcr_img * self.SIZE_TEMP + 48
                yc = ycr_img * self.SIZE_TEMP + 48
            
                ### Compute relative coord to 448x448 img and handle cropping
                xcr_img = xc/self.SIZE - posXcrop_rimg
                ycr_img = yc/self.SIZE - posYcrop_rimg

                ### Restrict the cropping btw 0 & 1
                xcr_img = np.clip(xcr_img, 0, 1)
                ycr_img = np.clip(ycr_img, 0, 1)

                wr_img += 48 / 500
                hr_img += 48 / 500

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
            if self.CROP: #or self.HARDCROP:
                img_PIL, crop_infos = img_PIL

        img = self._transform(img_PIL)
        target = self._encode(img_path, crop_infos)

        return img, target


def get_training_dataset(BATCH_SIZE=16, **kwargs):
    """
    Loads and maps the training split of the dataset using the custom dataset class. 
    """
    dataset = MealtraysDataset(root="../../mealtray_dataset/dataset", **kwargs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

def get_validation_dataset(BATCH_SIZE=None, **kwargs):
    """
    Loads and maps the validation split of the datasetusing the custom dataset class. 
    """
    dataset = MealtraysDataset(root="../../mealtray_dataset/dataset", **kwargs)
    if BATCH_SIZE is None:
        BATCH_SIZE = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return dataloader
