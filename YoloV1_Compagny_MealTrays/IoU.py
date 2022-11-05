import torch
from configparser import ConfigParser
import os, sys

current_folder = os.path.dirname(locals().get("__file__"))
config_file = os.path.join(current_folder, "config.ini")
sys.path.append(config_file)

config = ConfigParser()
config.read("config.ini")
S = config.getint("MODEL", "GRID_SIZE")

def relative2absolute(box, N, cell_i, cell_j)->torch.Tensor:
    """
    Turns relative box infos into absolute coordinates 
    xmin, ymin, xmax, ymax.
    TODO -> expliquer les trucs de cell_ij = vecteur
    TODO -> rendre la fonction universelle pour pred & target
    Used by the NMS module.

    Args:
        box (torch.Tensor of shape (N,S,S,5))
            Predicted bounding boxes, outputs of the model
        cell_i (int or torch.Tensor (N))
            Cell i position(s). Could also be the cells which countain an object.
            In that case, 'cell_i' is a tensor of size (N). ### bizarre -> TODO
        cell_j (int or torch.Tensor (N))
            Cell j position(s) TODO

    Returns:
        box_prediction_absolute (torch.Tensor of shape (N,5))
            Contains the 4 predicted coordinates xmin, ymin, 
            xmax, ymax and confidence_score for each image.
    """
    assert len(box.shape) == 4, "Error: box_prediction is torch.Tensor of shape (N,S,S,5)"
    assert box.shape[-1] == 5, "Error: box_prediction should contain a unique box -> (N,S,S,5)"

    SIZEHW = 448
    CELL_SIZE = 1/S

    ### Absolute center coordinates (xcyc+cell_size)*ji
    xcr_cell, ycr_cell = box[N, cell_i, cell_j, 0:2].permute(1,0)
    xcr_img = xcr_cell * CELL_SIZE + cell_j * CELL_SIZE
    ycr_img = ycr_cell * CELL_SIZE + cell_i * CELL_SIZE
    
    ### Fill tensor with all S*S possible bounding boxes
    # Top left absolute coordinates
    wr_img, hr_img = box[N, cell_i, cell_j, 2:4].permute(1,0)
    xmin = (xcr_img - wr_img/2) * SIZEHW
    ymin = (ycr_img - hr_img/2) * SIZEHW

    confidence_score = box[N, cell_i, cell_j, 4]

    # Bottom right absolute coordinates
    xmax = xmin + wr_img*SIZEHW
    ymax = ymin + hr_img*SIZEHW

    xmin, ymin, xmax, ymax = xmin.floor(), ymin.floor(), xmax.floor(), ymax.floor()

    box_absolute = torch.stack((xmin, ymin, xmax, ymax, confidence_score), dim=-1)
    return box_absolute


def intersection_over_union(box_1:torch.Tensor, box_2:torch.Tensor)->torch.Tensor:
    """
    Compute IoU between 2 boxes.
    Boxes should be [xmin, ymin, xmax, ymax, _] with absolute coordinates.

    Args:
        box_1 (torch.Tensor of shape (N,4))
        box_2 (torch.Tensor of shape (N,4))

    Return:
        iou (torch.Tensor of shape (N,1))
    """
    assert box_1.shape[-1] == 4, "Error: box_1 is torch.Tensor of shape (N,4)"
    assert box_2.shape[-1] == 4, "Error: box_2 is torch.Tensor of shape (N,4)"

    xmin_1, ymin_1, xmax_1, ymax_1 = box_1[:,:4].permute(1,0)
    xmin_2, ymin_2, xmax_2, ymax_2 = box_2[:,:4].permute(1,0)

    smoothing_factor = 1e-10
    zero = torch.Tensor([0])
       
    ### x, y overlaps btw 1 and 2
    xmin_overlap = torch.maximum(xmin_1, xmin_2)
    xmax_overlap = torch.minimum(xmax_1, xmax_2)
    ymin_overlap = torch.maximum(ymin_1, ymin_2)
    ymax_overlap = torch.minimum(ymax_1, ymax_2)

    ### Areas
    box_true_area = (xmax_1 - xmin_1) * (ymax_1 - ymin_1)
    box_pred_area = (xmax_2 - xmin_2) * (ymax_2 - ymin_2)

    ### Intersection area and union area
    overlap_area = torch.maximum((xmax_overlap - xmin_overlap), zero) * \
        torch.maximum((ymax_overlap - ymin_overlap), zero)
    union_area = (box_true_area + box_pred_area) - overlap_area
    
    ### Compute IoU
    iou = (overlap_area + smoothing_factor) / (union_area + smoothing_factor)
    return iou