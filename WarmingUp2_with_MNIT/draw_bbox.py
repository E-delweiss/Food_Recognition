from configparser import ConfigParser

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from icecream import ic

from smallnet import netMNIST
from validation import validation_loop
from MNIST_dataset import get_validation_dataset
import IoU
import utils
import NMS

#### TODO

S = 6
C = 10
B = 2
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
    # model = resnet(pretrained=True, in_channels=IN_CHANNEL, S=S, B=B, C=C)
    model = yoloResnet(load_yoloweights=True, pretrained=False, S=S, B=B, C=C).to(torch.device("mps"))
    
    print("Validation loop")
    validation_dataset = get_validation_dataset(BATCH_SIZE=32, isNormalize=True, isAugment=False)
    img, target, prediction = validation_loop(model, validation_dataset, device=torch.device("mps"), ONE_BATCH=True)

    draw_boxes(img, target, prediction)











#############################


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


S=6
B=2
C=10
sizeHW=140

model = netMNIST(load_weights=True, sizeHW=sizeHW, S=S, B=B, C=C).to(torch.device("mps"))

validation_dataset = get_validation_dataset(BATCH_SIZE=32)
# training_dataset = get_training_dataset(BATCH_SIZE=32)
img, target, prediction = validation_loop(model, validation_dataset, device=torch.device("mps"), ONE_BATCH=True)
# img, target, prediction = validation_loop(model, training_dataset, device=torch.device("mps"), ONE_BATCH=True)


### Choose image & rescale pixel values to un-normed values (from 0 -black- to 255 -white-)
BATCH_SIZE = len(img)
idx = np.random.randint(0, BATCH_SIZE)

img_idx = img[idx]
img_idx = img_idx * 255.0
img_idx = img_idx.to(torch.uint8)

# target_abs_box = IoU.relative2absolute(target[idx].unsqueeze(0))
# true_bboxes = utils.tensor2boxlist(target_abs_box, B=1, S=S, C=C)
# true_bboxes = [box for box in true_bboxes if box[4]>0]

# [x1, y1, w1, h1, c1, x2, y2, w2, h2, c1, C]
prediction = prediction[idx].unsqueeze(0)
prediction_temp = torch.concat((prediction[...,:5], prediction[...,5*B:]), dim=-1)
prediction_abs_box1 = IoU.relative2absolute(prediction_temp)
prediction_abs_box2 = IoU.relative2absolute(prediction[...,5:])

list_box1 = utils.tensor2boxlist(prediction_abs_box1, 1, S, C)
list_box2 = utils.tensor2boxlist(prediction_abs_box2, 1, S, C)
list_all_boxes = list_box1 + list_box2

bboxes_pred = [box[:4] for box in list_all_boxes if box[4] > 0.1]
labels_pred = [str(int(box[-1])) for box in list_all_boxes if box[4] > 0.99]
# ic(bboxes)


# true_bboxes_test = [box[:4] for box in true_bboxes]
# true_labels_test = [str(int(box[-1])) for box in true_bboxes]

draw_bbox = draw_bounding_boxes(image=img_idx, 
                                # boxes=torch.tensor(bboxes)[:,:4],
                                boxes=torch.tensor(bboxes_pred),
                                colors='red',
                                labels=labels_pred,
                                font_size=15,
                                font="Courier",
                                fill=True)

show(draw_bbox)

