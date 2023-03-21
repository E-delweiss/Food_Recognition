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
    img:torch.Tensor, target:torch.Tensor, prediction:torch.Tensor, frame_size, iou_threshold:float=0.5, prob_threshold:float=0.5):
    """
    TODO
    """
    ### Choose image & rescale pixel values to un-normed values (from 0 -black- to 255 -white-)
    BATCH_SIZE = len(img)
    idx = np.random.randint(0, BATCH_SIZE)

    img_idx = img[idx]
    img_idx = img_idx * 255.0
    img_idx = img_idx.to(torch.uint8)

    ### Choose image & rescale pixel values to un-normed values (from 0 -black- to 255 -white-)
    # img_idx = img[idx]
    # inv_normalize = torchvision.transforms.Normalize(
    #     mean=[-0.4168/0.3475, -0.4055/0.3442, -0.3838/0.3386],
    #     std=[1/0.3475, 1/0.3442, 1/0.3386]
    #     )
    # img_idx = inv_normalize(img_idx) * 255.0
    # img_idx = img_idx.to(torch.uint8)

    ### Extract target bounding boxes
    target_abs_box = IoU.relative2absolute(target[idx].unsqueeze(0), frame_size)
    true_bboxes = utils.tensor2boxlist(target_abs_box)
    true_bboxes = [box for box in true_bboxes if box[4]>0]

    ### Apply NMS
    nms_box_val = NMS.non_max_suppression(prediction[idx].unsqueeze(0), frame_size, prob_threshold, iou_threshold)
    
    all_pred_boxes = []
    for nms_box in nms_box_val:
        all_pred_boxes.append(nms_box)

    len_predbox = len(all_pred_boxes)
    len_truebox = len(true_bboxes)

    all_boxes = all_pred_boxes + true_bboxes

    labels_list = [str(int(a[-1])) for a in all_pred_boxes]
    labels_temp = [None for k in range(len_truebox)]
    labels = labels_list + labels_temp

    color_pred = ["red" for k in range(len_predbox)]
    color_true = ["white" for k in range(len_truebox)]
    colors = color_pred + color_true

    draw_bbox = draw_bounding_boxes(image=img_idx, 
                                    boxes=torch.tensor(all_boxes)[:,:4],
                                    labels=labels,
                                    colors=colors,
                                    font_size=15,
                                    font="Courier",
                                    fill=False)
    show(draw_bbox)


if __name__ == "__main__":
    S=6
    B=2
    C=10
    sizeHW=140
    model = netMNIST(load_weights=True, sizeHW=sizeHW, S=S, B=B, C=C).to(torch.device("mps"))
    
    print("Validation loop")
    validation_dataset = get_validation_dataset(BATCH_SIZE=512)
    img, target, prediction = validation_loop(model, validation_dataset, device=torch.device("mps"), ONE_BATCH=True)
    draw_boxes(img, target, prediction, frame_size=140)
