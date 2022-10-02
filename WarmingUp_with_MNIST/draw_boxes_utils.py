import numpy as np
import os
import PIL
import matplotlib.pyplot as plt

def draw_bounding_boxes_on_image_array(image:np.ndarray, box_true, box_pred, color:list=[], 
                                                                             thickness:int=1, display_str_list:list=[]):
    """Draws bounding boxes on image (numpy array).
        TODO : docstring correction
    Args:
        image: a numpy array object.
        ####boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
                     The coordinates are in normalized format between [0, 1].######
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        display_str_list_list: a list of strings for each bounding box.
    Raises:
        ValueError: if boxes is not a [N, 4] array
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
    draw_bounding_boxes_on_image(rgbimg, box_true, box_pred, color, thickness, display_str_list)
    plt.imshow(np.array(rgbimg))



def draw_bounding_boxes_on_image(image, box_true, box_pred, color_list:list=[], 
                                                                 thickness:int=1, display_str_list:list=[]):
    """Draws bounding boxes on image.
        TODO : doctring correction
    Args:
        image: PIL.Image.
        boxes: numpy array of shape (N,4)
            Contains (ymin, xmin, ymax, xmax). The coordinates are absolute.
        color: list, default is empty
            Color to draw bounding box.
        thickness: int, default value is 4
            Line thickness.
        display_str_list: tuple
            A list of strings for each bounding box.
                                                     
    Raises:
        ValueError: if boxes is not a [N, 4] array
    """
    draw_ONE_bounding_box_on_image(image, box_true[0], box_true[1], box_true[2], box_true[3], color=color_list[0], thickness=thickness)
    draw_ONE_bounding_box_on_image(image, box_pred[0], box_pred[1], box_pred[2], box_pred[3], color=color_list[1], thickness=thickness)

                                                            

def draw_ONE_bounding_box_on_image(image, xmin:float, ymin:float, xmax:float, ymax:float, 
                                                             color:list=[], thickness:int=1, display_str:list=[]):
    """Adds a bounding box to an image.
    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.
    
        TODO : doctring correction

    Args:
        image: a PIL.Image object.
        ymin: ymin of bounding box.
        xmin: xmin of bounding box.
        ymax: ymax of bounding box.
        xmax: xmax of bounding box.
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        display_str_list: string to display in box
        use_normalized_coordinates: If True (default), treat coordinates
            ymin, xmin, ymax, xmax as relative to the image.    Otherwise treat
            coordinates as absolute.
    """
    draw = PIL.ImageDraw.Draw(image)
    left, right, top, bottom = xmin, xmax, ymin, ymax
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)



