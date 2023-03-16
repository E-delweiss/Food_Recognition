import numpy as np
import random as rd

import torch
import torch.nn.functional as F
import torchvision


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, root:str, split:str="train", download:bool=False, S=6, sizeHW=75):
        if split == "test":
            train = False
        else:
            train = True
        
        self.B = 1
        self.C = 10
        self.S = S
        self.dataset = torchvision.datasets.MNIST(root=root, train=train, download=download)
        self.cell_size = 1/float(self.S)
    
    def __len__(self):
        return len(self.dataset)

    def _numpy_pad_to_bounding_box(self, idx_list, offset_height, offset_width, frame_height, frame_width):
        # assert image.shape[:-1][0] <= target_height-offset_height, "height must be <= target - offset"
        # assert image.shape[:-1][1] <= target_width-offset_width, "width must be <= target - offset"
        
        digit_size = 28
        
        images = []
        for idx in idx_list:
            img = torchvision.transforms.ToTensor()(self.dataset[idx][0])
            img = torch.reshape(img, (digit_size, digit_size, 1,))
            images.append(img)

        target_tensor = torch.zeros((frame_height, frame_width, 1))

        for k, img in enumerate(images):
            target_tensor[offset_height[k]:offset_height[k]+digit_size, offset_width[k]:offset_width[k]+digit_size] = img
        
        image = target_tensor.permute(2,0,1) #(C,H,W)
        image = image.to(torch.float)
        return image

    def _pasting(self, idx):
        
        digit_size = 28
        frame_size = 140
        
        xmin_list = []
        ymin_list = []
        xmin_list.append(rd.randint(0, frame_size-digit_size))
        ymin_list.append(rd.randint(0, frame_size-digit_size))

        for k in range(idx):
            area_of_intersection_list = [False]
            while all(area_of_intersection_list) != True:
                area_of_intersection_list = []
                xmin = rd.randint(0, frame_size-digit_size)
                ymin = rd.randint(0, frame_size-digit_size)
                xmax = xmin + digit_size
                ymax = ymin + digit_size
                for xmin_k, ymin_k in zip(xmin_list, ymin_list):
                    ix1 = np.maximum(xmin_k, xmin)
                    iy1 = np.maximum(ymin_k, ymin)
                    ix2 = np.minimum(xmin_k+digit_size, xmax)
                    iy2 = np.minimum(ymin_k+digit_size, ymax)

                    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
                    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
                    area_of_intersection = i_height * i_width
                    area_of_intersection_list.append(area_of_intersection == 0)

            xmin_list.append(xmin)
            ymin_list.append(ymin)

            ################
            image = self._numpy_pad_to_bounding_box(idx, ymin, xmin, frame_size, frame_size)

            for xmin, ymin in zip(xmin_list, ymin_list):
                xmin, ymin = xmin.to(torch.float), ymin.to(torch.float)
                xmax_bbox, ymax_bbox = (xmin + digit_size), (ymin + digit_size)
                xmin_bbox, ymin_bbox = xmin, ymin
            
                w_bbox = xmax_bbox - xmin_bbox
                h_bbox = ymax_bbox - ymin_bbox

                box = [xmin, ymin, w_bbox, h_bbox]
                
        return image, box

    def _encode(self, box, label):    
        ### Absolute box infos
        xmin, ymin, w_bbox, h_bbox = box
        
        ### Relative box infos
        wr_img = w_bbox / 75
        hr_img = h_bbox / 75
        xr_min = xmin / 75
        yr_min = ymin / 75

        ### x and y box center coords
        xcr_img = (xr_min + wr_img/2)
        ycr_img = (yr_min + hr_img/2)

        ### Object grid location
        i = (xcr_img / self.cell_size).ceil() - 1.0
        j = (ycr_img / self.cell_size).ceil() - 1.0
        i, j = int(i), int(j)

        ### x & y of the cell left-top corner
        x0 = i * self.cell_size
        y0 = j * self.cell_size
        
        ### x & y of the box on the cell, normalized from 0.0 to 1.0.
        xcr_cell = (xcr_img - x0) / self.cell_size
        ycr_cell = (ycr_img - y0) / self.cell_size

        ### Label one-hot encoding
        one_hot_label = F.one_hot(torch.as_tensor(label, dtype=torch.int64), self.C)

        ### 4 coords + 1 conf + 10 classes
        box_target = torch.zeros(self.S, self.S, self.B+4)
        box_target[j, i, :5] = torch.Tensor([xcr_cell, ycr_cell, wr_img, hr_img, 1.])

        return box_target, one_hot_label

    def __getitem__(self, idx):
        # idx useless
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        nb_digit = rd.randint(1, 6) #from 1 to 5
        idx = rd.choices(range(len(self.dataset)), k=nb_digit)
        
        # img = self.dataset[idx][0]
        # label = self.dataset[idx][1]

        image, box = self._pasting(idx)
        box, one_hot_label = self._encode(box, label?)
        
        return image, box, one_hot_label

##################################################################################################

def get_training_dataset(BATCH_SIZE=64):
    """
    Loads and maps the training split of the dataset using the custom dataset class. 
    """
    dataset = MNISTDataset(root="data", split="train", download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

def get_validation_dataset(BATCH_SIZE = None):
    """
    Loads and maps the validation split of the datasetusing the custom dataset class. 
    """
    dataset = MNISTDataset(root="data", split="test", download=True)
    if BATCH_SIZE is None:
        BATCH_SIZE = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return dataloader


if __name__ == "__main__":
    from icecream import ic
    dataset = MNISTDataset(root="data", split="train", download=True)
    img, box, label = dataset[3]
    ic(img.shape)