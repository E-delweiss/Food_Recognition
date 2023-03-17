import numpy as np
import random as rd

import torch
import torch.nn.functional as F
import torchvision


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, root:str, split:str="train", download:bool=False, S=6):
        if split == "test":
            train = False
        else:
            train = True
        
        self.digit_size = 28
        self.frame_size = 140
        self.C = 10
        self.S = S
        self.dataset = torchvision.datasets.MNIST(root=root, train=train, download=download)
        self.CELL_SIZE = 1/float(self.S)

    
    def __len__(self):
        return len(self.dataset)

    def _past_to_frame(self, idx_list, offset_height, offset_width, frame_height, frame_width):
        """
        """           
        images = []
        for idx in idx_list:
            img = torchvision.transforms.ToTensor()(self.dataset[idx][0])
            img = torch.reshape(img, (self.digit_size, self.digit_size, 1,))
            images.append(img)

        target_tensor = torch.zeros((frame_height, frame_width, 1))

        for k, img in enumerate(images):
            target_tensor[offset_height[k]:offset_height[k]+self.digit_size, offset_width[k]:offset_width[k]+self.digit_size] = img
        
        image = target_tensor.permute(2,0,1) #(C,H,W)
        image = image.to(torch.float)
        return image

    def _generate_target(self, idx_list): 
        """
        _summary_

        Args:
            idx_list (_type_): _description_

        Returns:
            _type_: _description_
        """
        xmin_list = []
        ymin_list = []
        box_list = []

        ### Find x, y coordinates for 'idx_list' digits that will not overlaps
        for k in range(len(idx_list)):
            area_of_intersection_list = [False]
            while all(area_of_intersection_list) != True:
                area_of_intersection_list = []
                xmin = rd.randint(0, self.frame_size-self.digit_size)
                ymin = rd.randint(0, self.frame_size-self.digit_size)
                xmax = xmin + self.digit_size
                ymax = ymin + self.digit_size
                for xmin_k, ymin_k in zip(xmin_list, ymin_list):
                    ix1 = np.maximum(xmin_k, xmin)
                    iy1 = np.maximum(ymin_k, ymin)
                    ix2 = np.minimum(xmin_k+self.digit_size, xmax)
                    iy2 = np.minimum(ymin_k+self.digit_size, ymax)

                    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
                    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
                    area_of_intersection = i_height * i_width
                    area_of_intersection_list.append(area_of_intersection == 0)

            xmin_list.append(xmin)
            ymin_list.append(ymin)

        ### Built box list such as box = [label, xmin, ymin, w_bbox, h_bbox]
        for k, (xmin, ymin) in enumerate(zip(xmin_list, ymin_list)):
            xmax_bbox, ymax_bbox = (xmin + self.digit_size), (ymin + self.digit_size)
            xmin_bbox, ymin_bbox = xmin, ymin

            w_bbox = xmax_bbox - xmin_bbox
            h_bbox = ymax_bbox - ymin_bbox

            label = self.dataset[idx_list[k]][1]
            box = [label, xmin, ymin, w_bbox, h_bbox]
            box_list.append(box)

        ### Past each digit to the main frame size (frame_size x frame_size)
        image = self._past_to_frame(idx_list, ymin_list, xmin_list, self.frame_size, self.frame_size)

        return image, box_list

    def _encode(self, list_box_target):
            """
            Encode box informations (coordinates, size and label) as a 
            (S,S,C+5) tensor.
            Args:
                img_path (str)
                    Absolute path of image.jpg
            Returns:
                target: torch.Tensor of shape (S,S,14)
                    Tensor of zeros representing the SxS grid. Zeros everywhere
                    except in the (j,i) positions where there is an object.
            """
            target = torch.zeros(self.S, self.S, self.C + 5)
            for box_infos in list_box_target:
                ### Absolute box infos
                label, xmin, ymin, w_bbox, h_bbox = box_infos

                ### Relative box infos
                wr_img = w_bbox / self.frame_size
                hr_img = h_bbox / self.frame_size
                xr_min = xmin / self.frame_size
                yr_min = ymin / self.frame_size

                ### x and y box center coords
                xcr_img = (xr_min + wr_img/2)
                ycr_img = (yr_min + hr_img/2)

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
                target[j, i, :5] = torch.Tensor([xcr_cell, ycr_cell, wr_img, hr_img, 1.])
                target[j, i, 5:] = one_hot_label
            return target


    def __getitem__(self, idx):      
        nb_digit = rd.randint(1, 6) #from 1 to 5
        idx = rd.choices(range(len(self.dataset)), k=nb_digit)
        
        image, box = self._generate_target(idx)
        target = self._encode(box)
        
        return image, target

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


# if __name__ == "__main__":
#     from icecream import ic
#     dataset = MNISTDataset(root="data", split="train", download=True)
#     img, box = dataset[3]
#     dataloader = get_training_dataset()
#     img = next(iter(dataloader))[0][8]
#     img_PIL = torchvision.transforms.ToPILImage()(img)
#     img_PIL.show()