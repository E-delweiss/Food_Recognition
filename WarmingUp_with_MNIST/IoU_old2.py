import torch

from utils import device


def relative2absolute_pred(box_pred_rel)->tuple:
    """
    Only for predicted bounding boxes. 
    It turns all relative-to-cell coords into absolute coords without
    considering objects.
    Later with NMS, those bounding boxes will be filter.

    Args:
        box_pred_rel : torch.Tensor of shape (N, S, S, 5)
            Bounding box coordinates to convert. xy are relative-to-cell 
            and wh are relative to image size.
    Return:
        box_absolute : torch.Tensor of shape (N,S*S,4)
            Contains the 4 coordinates xmin, ymin, xmax, ymax of all S*S 
            bounding boxes of each image.
    """
    # assert len(box_true.shape)==4 and len(box_pred.shape)==4, "Bbox should be of size (N,S,S,5)."

    SIZEHW = 75
    S = 6
    CELL_SIZE = 1/S
    BATCH_SIZE = len(box_pred_rel)
    
    xmin = torch.zeros((BATCH_SIZE, S*S))
    ymin = torch.zeros((BATCH_SIZE, S*S))
    xmax = torch.zeros((BATCH_SIZE, S*S))
    ymax = torch.zeros((BATCH_SIZE, S*S))

    it = 0
    for cell_i in range(S):
        for cell_j in range(S):
            ### Absolute center coordinates (xcyc+cell_size)*ji
            xcr_cell, ycr_cell = box_pred_rel[:,cell_i, cell_j, 0:2].permute(1,0)
            xcr_img = xcr_cell * CELL_SIZE + cell_j * CELL_SIZE
            ycr_img = ycr_cell * CELL_SIZE + cell_i * CELL_SIZE
            
            ### Fill tensor with all S*S possible bounding boxes
            # Top left absolute coordinates
            wr_img, hr_img = box_pred_rel[:,cell_i, cell_j, 0:2].permute(1,0)
            xmin[:, it] = (xcr_img - wr_img/2) * SIZEHW
            ymin[:, it] = (ycr_img - hr_img/2) * SIZEHW
            
            # Bottom right absolute coordinates
            xmax[:, it] = xmin[:, it] + wr_img*SIZEHW
            ymax[:, it] = ymin[:, it] + hr_img*SIZEHW
            it += 1

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

    box_absolute = torch.stack((xmin, ymin, xmax, ymax), dim=-1)
    return box_absolute


def intersection_over_union(box_true:torch.Tensor, box_pred:torch.Tensor, S:int=6)->torch.Tensor:
    """
    ?????

    Args:
        box_true : torch.Tensor of shape (N,S,S,5)
            ???
        box_pred : torch.Tensor of shape (N,S,S,5)
            Output prediction of the model.
        S (int, optional)
            Grid size. Defaults to 7.

    Return:
        all_iou : torch.Tensor of shape (N,S*S)
            ???
    """
    # assert box_true.shape[-1] == 5 and box_pred.shape[-1] == 5, "All bbox should be of shape (N,S,S,5)."

    nb_box_per_img = S*S
    BATCH_SIZE = len(box_pred)
    N = range(BATCH_SIZE)

    ### Convert cell reltative coordinates to absolute coordinates
    box_true = relative2absolute_true(box_true)
    box_pred = relative2absolute_pred(box_pred)

    xmin_true, ymin_true, xmax_true, ymax_true = box_true.permute(1,0)

    all_iou = torch.zeros(BATCH_SIZE, nb_box_per_img)
    smoothing_factor = 1e-10
    zero = torch.Tensor([0])
    for k in range(nb_box_per_img):
        ### Retrieve each coords. (N,36,4) -> 4*(N,1)
        xmin_pred, ymin_pred, xmax_pred, ymax_pred = box_pred[N,k,:].permute(1,0)
        
        ### x, y overlaps btw pred and groundtruth for the box_pred 'k': 
        xmin_overlap = torch.maximum(xmin_true, xmin_pred)
        xmax_overlap = torch.minimum(xmax_true, xmax_pred)
        ymin_overlap = torch.maximum(ymin_true, ymin_pred)
        ymax_overlap = torch.minimum(ymax_true, ymax_pred)
    
        ### Pred areas for the 'k' box_pred & true area
        box_true_area = (xmax_true - xmin_true) * (ymax_true - ymin_true)
        box_pred_area = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred)

        ### Compute intersection area, union area and IoU
        overlap_area = torch.maximum((xmax_overlap - xmin_overlap), zero) * \
            torch.maximum((ymax_overlap - ymin_overlap), zero)
        union_area = (box_true_area + box_pred_area) - overlap_area
        iou = (overlap_area + smoothing_factor) / (union_area + smoothing_factor)
    
    ### Set IoU to zero when there is no coordinates (i.e. no object)
    # iou = torch.masked_fill(iou, noObject, 0)
        all_iou[:, k] = iou

    return all_iou   



def test():
    S = 6
    box_true = torch.randint(0, 2, (16, S, S, 5))
    # box_pred = torch.rand(16, S, S, 5)
    # iou = intersection_over_union(box_true, box_pred)

if __name__ == '__main__':
    test()
