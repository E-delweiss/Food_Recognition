import torch

from utils import device


def relative2absolute(box_true:torch.Tensor, box_pred:torch.Tensor)->tuple:
    """
    Turns bounding box relative to cell coordinates into absolute coordinates 
    (pixels). Used to calculate IoU. 

    Args:
        box_true : torch.Tensor of shape (N, S, S, 5)
            Groundtruth bounding box coordinates to convert.
        box_pred : torch.Tensor of shape (N, S, S, 5)
            Predicted bounding box coordinates to convert.
    Return:
        box_true_absolute : torch.Tensor of shape (N, 4)
        box_pred_absolute : torch.Tensor of shape (N, 4)
    """
    assert len(box_true.shape)==4 and len(box_pred.shape)==4, "Bbox should be of size (N,S,S,5)."

    SIZEHW = 75
    S = 6
    CELL_SIZE = 1/S

    ### Get non-zero coordinates
    cells_with_obj = box_true.nonzero()[::5]
    N, cells_i, cells_j, _ = cells_with_obj.permute(1,0)

    ### Retrieving box coordinates. TBM if nb_obj > 1
    xrcell_true, yrcell_true, rw_true, rh_true = box_true[N, cells_i, cells_j, 0:4].permute(1,0)
    xrcell_pred, yrcell_pred, rw_pred, rh_pred = box_pred[N, cells_i, cells_j, 0:4].permute(1,0)   
    
    ### Compute relative-to-image center coordinates
    xc_rimg_true =  xrcell_true * CELL_SIZE + cells_j * CELL_SIZE
    xc_rimg_pred =  xrcell_pred * CELL_SIZE + cells_j * CELL_SIZE
    yc_rimg_true =  yrcell_true * CELL_SIZE + cells_i * CELL_SIZE
    yc_rimg_pred =  yrcell_pred * CELL_SIZE + cells_i * CELL_SIZE

    ### Compute absolute top left coordinates
    xmin_true = (xc_rimg_true - rw_true/2) * SIZEHW
    xmin_pred = (xc_rimg_pred - rw_pred/2) * SIZEHW
    ymin_true = (yc_rimg_true - rh_true/2) * SIZEHW
    ymin_pred = (yc_rimg_pred - rh_pred/2) * SIZEHW

    ### Compute absolute bottom right coordinates
    xmax_true = xmin_true + rw_true*SIZEHW 
    xmax_pred = xmin_pred + rw_pred*SIZEHW 
    ymax_true = ymin_true + rh_true*SIZEHW
    ymax_pred = ymin_pred + rh_pred*SIZEHW 

    ### Stacking
    box_true_absolute = torch.stack((xmin_true, ymin_true, xmax_true, ymax_true), dim=-1)
    box_pred_absolute = torch.stack((xmin_pred, ymin_pred, xmax_pred, ymax_pred), dim=-1)
    
    return box_true_absolute, box_pred_absolute




def intersection_over_union(box_true:torch.Tensor, box_pred:torch.Tensor)->torch.Tensor:
    """
    Intersection over Union method.

    Args:
        box_true : torch.Tensor of shape (N, S, S, 5)
            Bounding boxes of a batch, in a given cell.
        box_pred : torch.Tensor of shape (N, S, S, 5)
            Bounding boxes of a batch, in a given cell.

    Return:
        iou : torch.Tensor of shape (N,)
            Batch of floats between 0 and 1 where 1 is a perfect overlap.
    """
    assert box_true.shape[-1] == 5 and box_pred.shape[-1] == 5, "All bbox should be of shape (N,S,S,5)."

    ### Convert cell reltative coordinates to absolute coordinates: (N,S,S,5)->(N,4)
    box_true, box_pred = relative2absolute(box_true, box_pred)
    xmin_true, ymin_true, xmax_true, ymax_true = box_true.permute(1,0)
    xmin_pred, ymin_pred, xmax_pred, ymax_pred = box_pred.permute(1,0)

    ### There is no object if all coordinates are zero (TBM ?)
    noObject = (xmin_pred + ymin_pred + xmax_pred + ymax_pred).eq(0)

    smoothing_factor = 1e-10

    ### x, y overlaps btw pred and groundtruth: 
    xmin_overlap = torch.maximum(xmin_true, xmin_pred)
    xmax_overlap = torch.minimum(xmax_true, xmax_pred)
    ymin_overlap = torch.maximum(ymin_true, ymin_pred)
    ymax_overlap = torch.minimum(ymax_true, ymax_pred)
    
    ### Pred and groundtruth areas
    box1_area = (xmax_true - xmin_true) * (ymax_true - ymin_true)
    box2_area = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred)

    ### Compute intersection area, union area and IoU
    zero = torch.Tensor([0]).to(device())
    overlap_area = torch.maximum((xmax_overlap - xmin_overlap), zero) * \
        torch.maximum((ymax_overlap - ymin_overlap), zero)
    union_area = (box1_area + box2_area) - overlap_area
    iou = (overlap_area + smoothing_factor) / (union_area + smoothing_factor)
    
    ### Set IoU to zero when there is no coordinates (i.e. no object)
    iou = torch.masked_fill(iou, noObject, 0)

    return iou   



def test():
    S = 6
    box_true = torch.randint(0, 2, (16, S, S, 5))
    box_pred = torch.rand(16, S, S, 5)
    iou = intersection_over_union(box_true, box_pred)
    print(iou.shape)

if __name__ == '__main__':
    test()