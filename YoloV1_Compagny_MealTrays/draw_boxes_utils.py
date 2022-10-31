import os

import numpy as np
import PIL
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision

def draw_bounding_boxes_on_image_array(
    image:np.ndarray, box_true:np.ndarray, box_pred:np.ndarray, color:list=[],thickness:int=1
    ):
    """
    Draws bounding boxes on PIL image.
    TODO

    Args:
        image (np.ndarray of shape (N,1,75,75))
            Validation images reshaped and un-normalized. 
        box_true (np.ndarray of shape (N,4))
            Groundtruth boxes with absolute coordinates.
        box_pred (np.ndarray)
            Predicted boxes with absolute coordinates.
        color (list, optional)
            Color of bounding boxes. Defaults to [].
        thickness (int, optional)
            Thickness of lines. Defaults to 1.

    Returns:
        None.
    """
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

    image_pil = torchvision.transforms.ToPILImage()(image)

    ### Set plot configs
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image_pil)
    ax.set_xticks = []
    ax.set_yticks = []

    for k_box in range(len(box_true)):
        xmin = box_true[k_box, 0]
        ymin = box_true[k_box, 1]
        w = box_true[k_box, 2] - xmin
        h = box_true[k_box, 3] - ymin 

        # for k, _ in enumerate(box_pred):
        #     draw_ONE_bounding_box_on_image(image, box_pred[k,0], box_pred[k,1], box_pred[k,2], box_pred[k,3], color=color_list[1], thickness=thickness)


        color = color[0] #color_dict.get(label_dict.get(label))
        rect = patches.Rectangle((xmin, ymin), w, h, facecolor='none', edgecolor=color)
        ax.add_patch(rect)
        offset_x = 2
        offset_y = 10
        rect_txt = patches.Rectangle((xmin, ymin), w, 15, facecolor=color, edgecolor=color)
        ax.add_patch(rect_txt)
        # ax.text(xmin+offset_x, ymin+offset_y, label_dict.get(label), fontsize=8, color='w', family='monospace', weight='bold')



