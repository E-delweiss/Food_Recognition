import torch

from utils import device


def relative2absolute(box_rel)->tuple:
    """
    ????? 

    Args:
        box_rel : torch.Tensor of shape (N, S, S, 5)
            Bounding box coordinates to convert. xy are relative-to-cell 
            and wh are relative to image size.
    Return:
        box_absolute : torch.Tensor of shape (N,S*S) ?????
    """
    # assert len(box_true.shape)==4 and len(box_pred.shape)==4, "Bbox should be of size (N,S,S,5)."

    SIZEHW = 75
    S = 6
    CELL_SIZE = 1/S
    BATCH_SIZE = len(box_rel)
    
    xmin = torch.zeros((BATCH_SIZE, S*S))
    ymin = torch.zeros((BATCH_SIZE, S*S))
    xmax = torch.zeros((BATCH_SIZE, S*S))
    ymax = torch.zeros((BATCH_SIZE, S*S))

    it = 0
    for cell_i in range(S):
        for cell_j in range(S):
            ### Absolute center coordinates ji*(xy+cell_size)
            xc_rimg = box_rel[:,cell_i, cell_j, 0] * CELL_SIZE + cell_j * CELL_SIZE
            yc_rimg = box_rel[:,cell_i, cell_j, 1] * CELL_SIZE + cell_i * CELL_SIZE
            
            ### Top left absolute coordinates
            xmin[:, it] = (xc_rimg - box_rel[:,cell_i, cell_j, 2]/2) * SIZEHW
            ymin[:, it] = (yc_rimg - box_rel[:,cell_i, cell_j, 3]/2) * SIZEHW
            
            ### Bottom right absolute coordinates
            xmax[:, it] = xmin[:, it] + box_rel[:,cell_i, cell_j, 2]*SIZEHW
            ymax[:, it] = ymin[:, it] + box_rel[:,cell_i, cell_j, 3]*SIZEHW


    box_absolute = torch.stack((xmin, ymin, xmax, ymax), dim=-1)
    return box_absolute


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
    # box_pred = torch.rand(16, S, S, 5)
    # iou = intersection_over_union(box_true, box_pred)
    box_abs = relative2absolute(box_true)
    print(box_abs.shape)

if __name__ == '__main__':
    test()
