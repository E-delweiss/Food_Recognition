import torch

from utils import device


def relative2absolute(bbox_relative:torch.Tensor)->torch.Tensor:
    """
    Turns bounding box relative to cell coordinates into absolute coordinates 
    (pixels). Used to calculate IoU. 

    Args:
        bbox_relative : torch.Tensor of shape (N, S, S, 5)
            Bounding box coordinates to convert.
    Return:
        bbox_absolute : torch.Tensor of shape (N, 4)
    """
    assert len(bbox_relative.shape)==4, "Bbox should be of size (N,S,S,5)."

    SIZEHW = 75
    S = 6
    CELL_SIZE = 1/S

    cells_with_obj = bbox_relative.nonzero()[::5]
    N, cells_i, cells_j, _ = cells_with_obj.permute(1,0)

    ### Retrieving box coordinates. TBM if nb_obj > 1
    xrcell = bbox_relative[cells_i, cells_j,0]
    yrcell = bbox_relative[cells_i, cells_j,1]
    rw = bbox_relative[cells_i, cells_j,2]
    rh = bbox_relative[cells_i, cells_j,3]

    ### Absolute coordinates
    xmin = xrcell * cells_j * CELL_SIZE * SIZEHW
    ymin = yrcell * cells_i * CELL_SIZE * SIZEHW
    xmax = xmin + rw * SIZEHW
    ymax = ymin + rh * SIZEHW

    ### absolute center outputs
    # xc_abs = (xmin + xmax)/2
    # yc_abs = (ymin + ymax)/2

    bbox_absolute = torch.stack((xmin, ymin, xmax, ymax), dim=-1)
    return bbox_absolute




def intersection_over_union(boxe_true:torch.Tensor, box_pred:torch.Tensor)->torch.Tensor:
    """
    Intersection over Union method.

    Args:
        boxe_true : torch.Tensor of shape (N, S, S, 5)
            Bounding boxes of a batch, in a given cell.
        box_pred : torch.Tensor of shape (N, S, S, 5)
            Bounding boxes of a batch, in a given cell.

    Return:
        iou : torch.Tensor of shape (N,)
            Batch of floats between 0 and 1 where 1 is a perfect overlap.
    """
    assert boxe_true.shape[-1] >= 4 and box_pred.shape[-1] >= 4, "All bbox should be of shape (N,4) or (N,5)."

    ### Convert cell reltative coordinates to absolute coordinates: (N,S,S,5)->(N,4)
    boxe_true = relative2absolute(boxe_true)
    box_pred = relative2absolute(box_pred)
    xmin1, ymin1, xmax1, ymax1 = boxe_true.permute(1,0)
    xmin2, ymin2, xmax2, ymax2 = box_pred.permute(1,0)

    ### There is no object if all coordinates are zero
    noObject = (xmin2 + ymin2 + xmax2 + ymax2).eq(0)

    smoothing_factor = 1e-10

    ### x, y overlaps btw pred and groundtruth: 
    xmin_overlap = torch.maximum(xmin1, xmin2)
    xmax_overlap = torch.minimum(xmax1, xmax2)
    ymin_overlap = torch.maximum(ymin1, ymin2)
    ymax_overlap = torch.minimum(ymax1, ymax2)
    
    ### Pred and groundtruth areas
    box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)

    ### Compute intersection area, union area and IoU
    zero = torch.Tensor([0]).to(device())
    overlap_area = torch.maximum((xmax_overlap - xmin_overlap), zero) * torch.maximum((ymax_overlap - ymin_overlap), zero)
    union_area = (box1_area + box2_area) - overlap_area
    iou = (overlap_area + smoothing_factor) / (union_area + smoothing_factor)
    
    ### Set IoU to zero when there is no coordinates (i.e. no object)
    iou = torch.masked_fill(iou, noObject, 0)

    return iou   
