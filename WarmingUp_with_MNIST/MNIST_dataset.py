import numpy as np

import torch
import torch.nn.functional as F
import torchvision


class my_mnist_dataset(torch.utils.data.Dataset):
    def __init__(self, root:str, split:str="train", download:bool=False, S=6, sizeHW=75):
        if split == "test":
            train = False
        else:
            train = True
        
        self.C = 10
        self.S = S
        self.dataset = torchvision.datasets.MNIST(root=root, train=train, download=download)
        self.cell_size = 1/float(self.S)
    
    def __len__(self):
        return len(self.dataset)

    def _numpy_pad_to_bounding_box(self, image, offset_height=0, offset_width=0, target_height=0, target_width=0):
        assert image.shape[:-1][0] <= target_height-offset_height, "height must be <= target - offset"
        assert image.shape[:-1][1] <= target_width-offset_width, "width must be <= target - offset"
        
        target_array = np.zeros((target_height, target_width, image.shape[-1]))

        for k in range(image.shape[0]):
            target_array[offset_height+k][offset_width:image.shape[1]+offset_width] = image[k]
        
        return target_array

    def _pasting75(self, image):
        ### xmin, ymin of digit
        xmin = torch.randint(0, 48, (1,))
        ymin = torch.randint(0, 48, (1,))
        
        image = torchvision.transforms.ToTensor()(image)
        image = torch.reshape(image, (28,28,1,))
        image = torch.from_numpy(self._numpy_pad_to_bounding_box(image, ymin, xmin, 75, 75))
        image = image.permute(2, 0, 1) #(C,H,W)
        image = image.to(torch.float)
        
        xmin, ymin = xmin.to(torch.float), ymin.to(torch.float)

        xmax_bbox, ymax_bbox = (xmin + 28), (ymin + 28)
        xmin_bbox, ymin_bbox = xmin, ymin
        
        w_bbox = xmax_bbox - xmin_bbox
        h_bbox = ymax_bbox - ymin_bbox

        box = [xmin, ymin, w_bbox, h_bbox]
        return image, box

    def _encode(self, box, label):    
        ### Absolute box infos
        xmin, ymin, w_bbox, h_bbox = box
        
        ### Relative box infos
        rw = w_bbox / 75
        rh = h_bbox / 75
        rx_min = xmin / 75
        ry_min = ymin / 75

        ### x and y box center coords
        rxc = (rx_min + rw/2)
        ryc = (ry_min + rh/2)

        ### Object grid location
        i = (rxc / self.cell_size).ceil() - 1.0
        j = (ryc / self.cell_size).ceil() - 1.0
        i, j = int(i), int(j)

        ### x & y of the cell left-top corner
        x0 = i * self.cell_size
        y0 = j * self.cell_size
        
        ### x & y of the box on the cell, normalized from 0.0 to 1.0.
        x_norm = (rxc - x0) / self.cell_size
        y_norm = (ryc - y0) / self.cell_size

        ### 4 coords + 1 conf + 10 classes
        one_hot_label = F.one_hot(torch.as_tensor(label, dtype=torch.int64), 10)

        # box_target = torch.zeros(self.S, self.S, 4+1+self.C)
        box_target = torch.zeros(self.S, self.S, 4+1)
        box_target[j, i, :5] = torch.Tensor([x_norm, y_norm, rw, rh, 1.])
        # box_target[j, i, 5:] = one_hot_label

        return box_target, one_hot_label

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img = self.dataset[idx][0]
        label = self.dataset[idx][1]

        image, box = self._pasting75(img)
        box, one_hot_label = self._encode(box, label)
        
        return image, box, one_hot_label

##################################################################################################

def get_training_dataset(BATCH_SIZE=64):
    """
    Loads and maps the training split of the dataset using the custom dataset class. 
    """
    dataset = my_mnist_dataset(root="data", split="train", download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

def get_validation_dataset(BATCH_SIZE = None):
    """
    Loads and maps the validation split of the datasetusing the custom dataset class. 
    """
    dataset = my_mnist_dataset(root="data", split="test", download=True)
    if BATCH_SIZE is None:
        BATCH_SIZE = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return dataloader

def test():
    dataloader = get_training_dataset()
    imgs, boxes, labels = next(iter(dataloader))

    print("Image size : ", imgs.shape)
    print("Image type : ", type(imgs))
    print("Boxes size : ", boxes.shape)
    print("Boxes type : ", type(boxes))
    print("Label size : ", labels.shape)
    print("Label type : ", type(labels))

    print("One hot label : ", labels[np.random.randint(0,64)])

if __name__ == '__main__':
    test()