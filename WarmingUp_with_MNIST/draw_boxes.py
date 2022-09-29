import torch
import numpy as np
import matplotlib.pyplot as plt 
import os
import PIL

import IoU
import NMS
import plot_utils

############### Matplotlib config
plt.rc('image', cmap='gray')
plt.rc('grid', linewidth=0)
plt.rc('xtick', top=False, bottom=False, labelsize='large')
plt.rc('ytick', left=False, right=False, labelsize='large')
plt.rc('axes', facecolor='F8F8F8', titlesize="large", edgecolor='white')
plt.rc('text', color='a8151a')
plt.rc('figure', facecolor='F0F0F0')# Matplotlib fonts
MATPLOTLIB_FONT_DIR = os.path.join(os.path.dirname(plt.__file__), "mpl-data/fonts/ttf")
################################################################################

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
    n_imgs = n_imgs[indexes].numpy()
    n_imgs = n_imgs * 255.0
    n_imgs = n_imgs.reshape(nb_sample, 75, 75)

    ### Choose label & argmax of one-hot vectors. 
    n_label_pred = label_pred[indexes]
    n_label_pred = torch.argmax(torch.softmax(n_label_pred, dim=-1), dim=-1).numpy() #(N,S,S,10) -> (N,S,S,1)

    ### Groundtruth boxes & preds
    n_box_true = box_true[indexes]
    n_box_pred = box_pred[indexes]

    ### Turn to absolute coords
    n_box_true_abs = IoU.relative2absolute_true(n_box_true).numpy()
    tensor_box_pred_abs, temp_indices = NMS.non_max_suppression(n_box_pred)
    n_box_pred_abs = tensor_box_pred_abs[N, temp_indices[:,0], temp_indices[:,1]].numpy()

    # Set plot configs
    all_plots = plt.figure(figsize=(20, 4))
    plt.title(title)
    plt.yticks([])
    plt.xticks([])

    for k in N :
        subplots = all_plots.add_subplot(1, nb_sample, k+1)

        # Dispay label name (0,1,2,...,9)
        plt.xlabel(n_label_pred[k])
        
        # Display in red if pred is false
        if n_label_pred[k] != label_true[k]:
            subplots.xaxis.label.set_color('red')

        img_to_draw = plot_utils.draw_bounding_boxes_on_image_array(
            image = n_imgs[k], 
            box_true = n_box_true_abs[k], 
            box_pred = n_box_pred_abs[k],
            color = ["white", "red"], 
            display_str_list = ["true", "pred"]
            )

    ### TODO extract IoU from relative2absolute
    # if len(iou) > i :
    #     color = "black"
    # if (n_iou[i] < iou_threshold):
    #   color = "red"
    # ax.text(0.2, -0.3, "iou: %s" %(n_iou[i]), color=color, transform=ax.transAxes)
    
    plt.imshow(img_to_draw)
 
   




