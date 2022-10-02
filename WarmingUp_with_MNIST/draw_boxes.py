import torch
import numpy as np
import matplotlib.pyplot as plt 
import os

from Darknet_like import YoloMNIST
from Validation import validation_loop
from MNIST_dataset import get_validation_dataset
import IoU
import NMS
import draw_boxes_utils



def draw_boxes(img, label_true, label_pred, box_true, box_pred, 
                                            iou_threshold=0.6,
                                            nb_sample=10, 
                                            title="MNIST groundtruth vs predictions"):
    """
    TBM
    /!\ box_pred i.e. box_pred_NMS
    """
    BATCH_SIZE = len(img)
    indexes = np.random.choice(BATCH_SIZE, size=nb_sample)
    N = range(nb_sample)

    ### Choose image & rescale pixel values to un-normed values (from 0 -black- to 255 -white-)
    n_imgs = img[indexes].numpy()
    n_imgs = n_imgs * 255.0
    n_imgs = n_imgs.reshape(nb_sample, 75, 75)

    ### Choose label & argmax of one-hot vectors.
    n_label_true = label_true[indexes]
    n_label_true_argmax = torch.argmax(n_label_true, dim=-1).numpy()

    n_label_pred = label_pred[indexes]
    n_label_pred_argmax = torch.argmax(torch.softmax(n_label_pred, dim=-1), dim=-1).numpy() #(N,S,S,10) -> (N,S,S,1)

    ### Groundtruth boxes & preds
    n_box_true = box_true[indexes]
    n_box_pred = box_pred[indexes]

    ### Turn to absolute coords and get indices of box positions after NMS
    n_box_true_abs = IoU.relative2absolute_true(n_box_true).numpy()
    n_box_pred_abs, temp_indices = NMS.non_max_suppression(n_box_pred, n_label_pred, 0.6)
    n_box_pred_abs = n_box_pred_abs[N, temp_indices[:,0], temp_indices[:,1]].numpy()
    
    ### Get label predictions with respect to prediction boxes
    n_label_pred_argmax = n_label_pred_argmax[N, temp_indices[:,0], temp_indices[:,1]]

    ### Get IoU between pred and true boxes
    n_iou = IoU.intersection_over_union(n_box_true_abs, n_box_pred_abs)

    # Set plot configs
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
    model_MNIST = YoloMNIST(75, 6, 10, 1)
    model_MNIST.load_state_dict(torch.load("yolo_mnist_model_10epochs_relativeCoords_29092022_19h41.pt"))
    
    print("Validation loop")
    validation_dataset = get_validation_dataset(64)
    img, box_true, box_pred, label_true, label_pred = validation_loop(model_MNIST, validation_dataset)

    draw_boxes(img, label_true, label_pred, box_true, box_pred)
