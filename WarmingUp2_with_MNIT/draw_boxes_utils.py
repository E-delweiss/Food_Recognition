import numpy as np
import os
import PIL
import matplotlib.pyplot as plt

def draw_bounding_boxes_on_image_array(
    image:np.ndarray, box_true:np.ndarray, box_pred:np.ndarray, color:list=[],thickness:int=1
    ):
    """
    Draws bounding boxes on PIL image.

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

    plt.yticks([])
    plt.xticks([])

    image_pil = PIL.Image.fromarray(image)
    rgbimg = PIL.Image.new("RGBA", image_pil.size)
    rgbimg.paste(image_pil)
    draw_bounding_boxes_on_image(rgbimg, box_true, box_pred, color, thickness)
    plt.imshow(np.array(rgbimg))



def draw_bounding_boxes_on_image(
    image:PIL.Image, box_true:np.ndarray, box_pred:np.ndarray, color_list:list=[], thickness:int=1
    ):
    """
    Draws bounding boxes on image. Calls the module draw_ONE_bounding_box_on_image twice to plot 
    both groundtruth and predicted bounding boxes.

    Args:
        image (PIL.Image)
            PIL Image on which the 75x75 image has been pasted.
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
    draw_ONE_bounding_box_on_image(image, box_true[0], box_true[1], box_true[2], box_true[3], color=color_list[0], thickness=thickness)
    draw_ONE_bounding_box_on_image(image, box_pred[0], box_pred[1], box_pred[2], box_pred[3], color=color_list[1], thickness=thickness)

                                                            

def draw_ONE_bounding_box_on_image(
    image, xmin:float, ymin:float, xmax:float, ymax:float, color:list=[], thickness:int=1
    ):
    """
    Draw lines on PIL Image.
    
    Args:
        image (PIL.Image)
            PIL Image on which the 75x75 image has been pasted.
        xmin (float)
            Top left x value of bounding box.
        ymin (float)
            Top left y value of bounding box.
        xmax (float)
            Bottom right x value of bounding box.
        tmax (float)
            Bottom right y value of bounding box.
        color (list, optional)
            Color of bounding boxes. Defaults to [].
        thickness (int, optional)
            Thickness of lines. Defaults to 1.

    Returns:
        None
    """
    draw = PIL.ImageDraw.Draw(image)
    left, right, top, bottom = xmin, xmax, ymin, ymax
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)



