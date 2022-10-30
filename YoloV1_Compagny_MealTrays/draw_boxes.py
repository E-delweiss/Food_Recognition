import os
from configparser import ConfigParser

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from YoloV1_Compagny_MealTrays.train import IN_CHANNEL 

from resnet101 import YoloV1
from validation import validation_loop
from mealtrays_dataset import get_validation_dataset
import IoU
import utils
# import NMS
import draw_boxes_utils

config = ConfigParser()
config.read("config.ini")
IN_CHANNEL = config.getint("MODEL", "in_channel")
S = config.getint("MODEL", "GRID_SIZE")
C = config.getint("MODEL", "NB_CLASS")
B = config.getint("MODEL", "NB_BOX")

def draw_boxes(
    img:torch.Tensor, target:torch.Tensor, prediction:torch.Tensor, iou_threshold:float=0.6, nb_sample:int=10, title:str=""
    ):
    """
    Display 'n' images and draw groundtruth and predicted bounding boxes on it.
    Show predicted class and IoU value between the two boxes.

    Args:
        img (torch.Tensor of shape (N,1,75,75))
            Images from validation dataset.
        TODO
        iou_threshold (float, optional)
            Defaults to 0.6.
        nb_sample (int, optional)
            Nb of sample to display. Defaults to 10.
        title (str, optional)
            Title of the plot. Defaults to "".

    Returns:
        None
    """

    BATCH_SIZE = len(img)
    indexes = np.random.choice(BATCH_SIZE, size=nb_sample)
    N = range(nb_sample)

    ### Choose image & rescale pixel values to un-normed values (from 0 -black- to 255 -white-)
    n_imgs = img[indexes]
    # mean=(0.4168, 0.4055, 0.3838), std=(0.3475, 0.3442, 0.3386)
    inv_normalize = torchvision.transforms.Normalize(
        mean=[-0.4168/0.3475, -0.4055/0.3442, -0.3838/0.3386],
        std=[1/0.3475, 1/0.3442, 1/0.3386]
        )
    n_imgs = inv_normalize(n_imgs).numpy()
    n_imgs = n_imgs * 255.0
    # n_imgs = n_imgs.reshape(nb_sample, 3, 448, 448)

    ### Choose label & argmax of one-hot vectors.
    n_label_true = target[indexes,:,:,5:]
    n_label_true_argmax = torch.argmax(n_label_true, dim=-1).numpy()

    n_label_pred = prediction[indexes,:,:,10:]
    n_label_pred_argmax = torch.argmax(torch.softmax(n_label_pred, dim=-1), dim=-1).numpy() #(N,S,S,8) -> (N,S,S,1)

    ### Groundtruth & preds boxes
    n_box_true = target[indexes, :, :, :5]
    n_box_pred = prediction[indexes, :, :, :10] ### !!!

#### ------------------------ ##########

    ### Turn to absolute coords and get indices of box positions after NMS
    N, cell_i, cell_j = utils.get_cells_with_object(target)
    n_box_true_abs = IoU.relative2absolute(n_box_true, N, cell_i, cell_j)

    iou_box = []
    for b in range(B):
        box_k = 5*b
        n_box_pred = IoU.relative2absolute(n_box_pred[:,:,:, box_k : 5+box_k], N, cell_i, cell_j) # -> (N,4)
        iou = IoU.intersection_over_union(n_box_true_abs, n_box_pred) # -> (N,1)
        iou_box.append(iou) # -> [iou_box1:(N), iou_box:(N)]    

    iou_mask = torch.gt(iou_box[0], iou_box[1])
    iou_mask = iou_mask.to(torch.int64)
    idx = 5*iou_mask #if 0 -> box1 infos, if 5 -> box2 infos
    x_pred_abs = n_box_pred[N, cell_i, cell_j, idx]                
    y_pred_abs = n_box_pred[N, cell_i, cell_j, idx+1]                
    w_pred_abs = n_box_pred[N, cell_i, cell_j, idx+2]
    h_pred_abs = n_box_pred[N, cell_i, cell_j, idx+3] ## ???

    n_box_pred_abs = torch.stack((x_pred_abs, y_pred_abs, w_pred_abs, h_pred_abs), dim=-1)


    n_box_pred_abs, temp_indices = NMS.non_max_suppression(n_box_pred, n_label_pred, 0.6)
    n_box_pred_abs = n_box_pred_abs[N, temp_indices[:,0], temp_indices[:,1]]
    
    ### Get label predictions with respect to prediction boxes
    n_label_pred_argmax = n_label_pred_argmax[N, temp_indices[:,0], temp_indices[:,1]]

    ### Get IoU between pred and true boxes
    n_iou = IoU.intersection_over_union(n_box_true_abs, n_box_pred_abs)

    ### draw_boxes_utils uses arrays only
    n_box_true_abs = n_box_true_abs.numpy()
    n_box_pred_abs = n_box_pred_abs.numpy()

    ### Set plot configs
    all_plots = plt.figure(figsize=(20, 4))
    plt.title(title, fontweight='bold')
    plt.yticks([])
    plt.xticks([])

    print("Draw boxes...")
    for k in N :
        subplots = all_plots.add_subplot(1, nb_sample, k+1)

        # Dispay label name (0,1,2,...,9)
        plt.xlabel(f"Digit : {n_label_pred_argmax[k]}", fontweight='bold')
        
        # Display in red if pred is false
        if n_label_pred_argmax[k] != n_label_true_argmax[k]:
            subplots.xaxis.label.set_color('red')
        else : 
            subplots.xaxis.label.set_color('green')
        
        # Draw true and pred bounding boxes
        img_to_draw = draw_boxes_utils.draw_bounding_boxes_on_image_array(
            image = n_imgs[k], 
            box_true = n_box_true_abs[k], 
            box_pred = n_box_pred_abs[k],
            color = ["white", "red"], 
            display_str_list = ["true", "pred"]
            )

        # Display iou in red if iou < threshold
        if (n_iou[k] < iou_threshold):
            color = "red"
        else :
            color = "green"
        subplots.text(0.10, -0.4, f"iou: {n_iou[k].item():.3f}", color=color, transform=subplots.transAxes, fontweight='bold')
            
    plt.show()
 
   

if __name__ == "__main__":
    os.chdir("WarmingUp_with_MNIST")

    print("Load model...")
    resnet101 = YoloV1(IN_CHANNEL, S, C, B)
    resnet101.load_state_dict(torch.load("yoloPlato_resnet101_150epochs_25102022_06h17.pt"))
    
    print("Validation loop")
    validation_dataset = get_validation_dataset(64)
    img, target, prediction = validation_loop(resnet101, validation_dataset)

    draw_boxes(img, label_true, label_pred, box_true, box_pred, title="MNIST groundtruth vs predictions")
