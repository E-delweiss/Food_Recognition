import torch

from utils import device

def relative2absolute_pred(box_pred_rel, cell_i, cell_j)->tuple:
    """
    box_pred_rel shape (N,5) ???
    [x, y, w , h]
    TODO
    """
    # assert len(box_true.shape)==4 and len(box_pred.shape)==4, "Bbox should be of size (N,S,S,5)."

    SIZEHW = 75
    S = 6
    CELL_SIZE = 1/S
    BATCH_SIZE = len(box_pred_rel)
    N = range(BATCH_SIZE)
    
    ### Absolute center coordinates (xcyc+cell_size)*ji
    xcr_cell, ycr_cell = box_pred_rel[N, cell_i, cell_j, 0:2].permute(1,0)
    xcr_img = xcr_cell * CELL_SIZE + cell_j * CELL_SIZE
    ycr_img = ycr_cell * CELL_SIZE + cell_i * CELL_SIZE
    
    ### Fill tensor with all S*S possible bounding boxes
    # Top left absolute coordinates
    wr_img, hr_img = box_pred_rel[N,cell_i, cell_j, 2:4].permute(1,0)
    xmin = (xcr_img - wr_img/2) * SIZEHW
    ymin = (ycr_img - hr_img/2) * SIZEHW
    
    # Bottom right absolute coordinates
    xmax = xmin + wr_img*SIZEHW
    ymax = ymin + hr_img*SIZEHW

    xmin, ymin, xmax, ymax = xmin.floor(), ymin.floor(), xmax.floor(), ymax.floor()

    box_absolute = torch.stack((xmin, ymin, xmax, ymax), dim=-1)
    return box_absolute


def relative2absolute_true(box_true_rel)->tuple:
    """
    Only for groundtruth bounding boxes.
    Each groundtruth bboxes is a zero (N,S,S,5) tensor except at the
    ji position where there are the corresponding bounding boxe coordinates. 
    
    Args:
        box_rel : torch.Tensor of shape (N, S, S, 5)
            Bounding box coordinates to convert. xy are relative-to-cell 
            and wh are relative to image size.
    Return:
        box_absolute : torch.Tensor of shape (N,4)
            Contains the 4 coordinates xmin, ymin, xmax, ymax for each image.
    """
    # assert len(box_true.shape)==4 and len(box_pred.shape)==4, "Bbox should be of size (N,S,S,5)."

    SIZEHW = 75
    S = 6
    CELL_SIZE = 1/S

    ### Get non-zero ij coordinates
    cells_with_obj = box_true_rel.nonzero()[::5]
    N, cells_i, cells_j, _ = cells_with_obj.permute(1,0)

    ### Retrieve box coordinates. TBM if nb_obj > 1
    xr_cell, yr_cell, wr_img, hr_img = box_true_rel[N, cells_i, cells_j, 0:4].permute(1,0)

    ### Compute xc and yc center coordinates relative to the image
    xcr_img =  xr_cell * CELL_SIZE + cells_j * CELL_SIZE
    ycr_img =  yr_cell * CELL_SIZE + cells_i * CELL_SIZE

    ### Compute absolute top left  and bottom right coordinates
    xmin = (xcr_img - wr_img/2) * SIZEHW
    ymin = (ycr_img - hr_img/2) * SIZEHW
    xmax = xmin + wr_img*SIZEHW 
    ymax = ymin + hr_img*SIZEHW

    xmin, ymin, xmax, ymax = xmin.floor(), ymin.floor(), xmax.floor(), ymax.floor()

    box_absolute = torch.stack((xmin, ymin, xmax, ymax), dim=-1, )
    return box_absolute


def intersection_over_union(box_1:torch.Tensor, box_2:torch.Tensor)->float:
    """
    Compute IoU between 2 boxes.
    Boxes should be [xmin, ymin, xmax, ymax, _] with absolute coordinates.

    Args:
        box_1 (torch.Tensor of shape (N,4) or (N,5))
        box_2 (torch.Tensor of shape (N,4) or (N,5))

    Return:
        iou (torch.Tensor of shape (N,1))
    """
    assert len(box_1.shape) == 2 and len(box_2.shape) == 2, "Error shape."
    box_1, box_2 = torch.Tensor(box_1), torch.Tensor(box_2)

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

    ### Intersection area, union area and IoU
    overlap_area = torch.maximum((xmax_overlap - xmin_overlap), zero) * \
        torch.maximum((ymax_overlap - ymin_overlap), zero)
    union_area = (box_true_area + box_pred_area) - overlap_area
    iou = (overlap_area + smoothing_factor) / (union_area + smoothing_factor)

    return iou