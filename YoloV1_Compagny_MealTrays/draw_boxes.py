from cProfile import label
from configparser import ConfigParser

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes

from darknet import YoloV1
from validation import validation_loop
from mealtrays_dataset import get_validation_dataset
import IoU
import utils
# import NMS

config = ConfigParser()
config.read("config.ini")
IN_CHANNEL = config.getint("MODEL", "in_channel")
S = config.getint("MODEL", "GRID_SIZE")
C = config.getint("MODEL", "NB_CLASS")
B = config.getint("MODEL", "NB_BOX")


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

def draw_boxes(
    img:torch.Tensor, target:torch.Tensor, prediction:torch.Tensor, iou_threshold:float=0.6, nb_sample:int=3, title:str="", PRINT_ALL=True
    ):
    """
    TODO
    """
    color_dict = {'Assiette':'b', 'Entree':'g', 'Pain':'r', 'Boisson':'c', 
            'Yaourt':'darkred', 'Dessert':'k', 'Fruit':'m', 'Fromage':'y'}
    label_dict = {0:'Assiette', 1:'Entree', 2:'Pain', 3:'Boisson', 
        4:'Yaourt', 5:'Dessert', 6:'Fruit', 7:'Fromage'}

    BATCH_SIZE = len(img)
    idx = np.random.randint(0, BATCH_SIZE)

    ### Choose image & rescale pixel values to un-normed values (from 0 -black- to 255 -white-)
    img_idx = img[idx]
    inv_normalize = torchvision.transforms.Normalize(
        mean=[-0.4168/0.3475, -0.4055/0.3442, -0.3838/0.3386],
        std=[1/0.3475, 1/0.3442, 1/0.3386]
        )
    img_idx = inv_normalize(img_idx) * 255.0
    img_idx = img_idx.to(torch.uint8)

    ### Choose label & argmax of one-hot vectors.
    label_true = target[idx,:,:,5:]
    label_true_argmax = torch.argmax(label_true, dim=-1)

    label_pred = prediction[idx,:,:,10:]
    label_pred_argmax = torch.argmax(torch.softmax(label_pred, dim=-1), dim=-1) #(N,S,S,8) -> (N,S,S,1)

    ### Groundtruth & preds boxes
    box_true = target[idx, :, :, :5].unsqueeze(0)
    box_pred = prediction[idx, :, :, :10] ### !!!

#### ------------------------ ##########

    ### Get cells with object
    N, cell_i, cell_j = utils.get_cells_with_object(box_true)
    
    ### Convert to absolute coord
    box_true_abs = IoU.relative2absolute(box_true, N, cell_i, cell_j)

    ### Retrive label names as a list 
    label_true_argmax = label_true_argmax[cell_i, cell_j]
    label_list = [label_dict.get(label.item()) for label in label_true_argmax]

    ### Choose the best predicted bounding box
    #NMS !!!

    # n_box_pred_abs = IoU.relative2absolute(box_pred[:,:,:,:5], N, cell_i, cell_j) #TBM
    N_unique = torch.unique(N)
    N_freq = torch.bincount(N)
    N_cumsum = torch.cumsum(N_freq, dim=0)
    N_sort = torch.stack((N_unique, N_freq, N_cumsum), dim=-1)

    start = N_sort[0,2]-N_sort[0,1]
    stop = N_sort[0,2]
    drawn_boxes = draw_bounding_boxes(img_idx.to(torch.uint8), box_true_abs[start: stop], labels=label_list, width=2, font='Arial Bold', font_size=15)
    show(drawn_boxes)
   

if __name__ == "__main__":
    print("Load model...")
    darknet = YoloV1(pretrained=True, in_channels=IN_CHANNEL, S=S, B=B, C=C)
    
    print("Validation loop")
    validation_dataset = get_validation_dataset(8, isNormalize=True, isAugment=False)
    img, target, prediction = validation_loop(darknet, validation_dataset, ONE_BATCH=True)

    draw_boxes(img, target, prediction)
