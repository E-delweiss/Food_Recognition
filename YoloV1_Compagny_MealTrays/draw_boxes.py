from configparser import ConfigParser

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from icecream import ic

from resnet50_old import resnet
from yoloResnet import yoloResnet
from validation import validation_loop
from mealtrays_dataset import get_validation_dataset
import IoU
import utils
import NMS

config = ConfigParser()
config.read("config.ini")
IN_CHANNEL = config.getint("MODEL", "in_channel")
S = config.getint("MODEL", "GRID_SIZE")
C = config.getint("MODEL", "NB_CLASS")
B = config.getint("MODEL", "NB_BOX")
PROB_THRESHOLD = 0.4
IOU_THRESHOLD = 0.2


def show(imgs):
    """
    TODO
    """
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
    color_dict = {'Plate':'blue', 'Starter':'green', 'Bread':'red', 'Drink':'cyan', 
            'Yogurt':'darkred', 'Dessert':'black', 'Fruit':'magenta', 'Cheese':'yellow'}
    label_dict = {0:'Plate', 1:'Starter', 2:'Bread', 3:'Drink', 
        4:'Yogurt', 5:'Dessert', 6:'Fruit', 7:'Cheese'}

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

    ### TODO
    target_abs_box = IoU.relative2absolute(target[idx].unsqueeze(0))
    true_bboxes = utils.tensor2boxlist(target_abs_box)
    true_bboxes = [box for box in true_bboxes if box[4]>0]
    bboxes_nms = NMS.non_max_suppression(prediction[idx].unsqueeze(0), PROB_THRESHOLD, IOU_THRESHOLD)

    labels_list = [label_dict.get(box[5]) for box in bboxes_nms]
    colors_list = [color_dict.get(label_name) for label_name in labels_list]

    draw_bbox = draw_bounding_boxes(image=img_idx, 
                                    boxes=torch.tensor(bboxes_nms)[:,:4],
                                    labels=labels_list,
                                    colors=colors_list,
                                    font_size=15,
                                    font="Courier",
                                    fill=True)
    show(draw_bbox)


if __name__ == "__main__":
    print("Load model...")
    model = resnet(pretrained=True, in_channels=IN_CHANNEL, S=S, B=B, C=C)
    # model = yoloResnet(load_yoloweights=True, pretrained=False, S=S, B=B, C=C)
    
    print("Validation loop")
    validation_dataset = get_validation_dataset(isNormalize=True, isAugment=False)
    img, target, prediction = validation_loop(model, validation_dataset, ONE_BATCH=True)

    draw_boxes(img, target, prediction)
