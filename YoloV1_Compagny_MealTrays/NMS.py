import torch

import IoU

def find_indices_max(box):
    """
    Retrieve columns and rows of max values.

    Arguments : 
        box : torch.Tensor of shape (N,S,S,1)
    
    Returns :
        batch_indices : torch.Tensor of shape (N,2)
    """
    N = box.shape[0]
    S = box.shape[1]

    ### Reshape (N,S,S) -> (N, S*S)
    box_reshape = box.view(N, S*S)
    indices = torch.argmax(box_reshape, dim=1)

    ### Compute columns and rows
    col_indices = (indices / S).to(torch.int32)
    row_indices = indices % S

    ### Stack and retrieve indices of max confidence number (N,2)
    batch_indices = torch.stack((col_indices, row_indices)).T
    return batch_indices


def non_max_suppression(box_pred, label_pred, iou_threshold):
    """
    Use confidence number and class probabilities such as pc = pc * P(C)
    to keep only predicted boxes with the highest confidence number.
    Convert boxes into absolute coordinates and compute IoU.

    Arguments :
        box_pred : torch.Tensor of shape (N,S,S,5)
            Predicted boxes
        label_pred : torch.Tensor of shape (N,S,S,10)
            Predicted labels
        iou_threshold : float TODO 
    """
    box = box_pred #(N,S,S,5)
    labels = label_pred #(N,S,S,10)
    
    BATCH_SIZE = len(box)
    S = 7
    N = range(BATCH_SIZE)

    ### Apply softmax to multi class predictions
    labels_prob = torch.softmax(labels, dim=-1)

    ### confidence number Pc is confidence number times class prediction
    box[:,:,:,4] = torch.mul(box[:,:,:,4], torch.amax(labels_prob, dim=-1))

    # 1) finding indices i,j of the max confidence number of each image in the batch
    m = find_indices_max(box[:,:,:,4])

    # 2) Getting boxes with the highest conf number for each image
    # box_max_confidence = box[N, m[:,0], m[:,1]] #(N,5)
    box_max_confidence = IoU.relative2absolute_pred(box, m[:,0], m[:,1]) #(N,5)

    # 3) Removing boxes with the highest pc numbers
    box[N, m[:,0], m[:,1]] = torch.Tensor([0])

    # 4) TBM with 2 bboxes and 1+ objects
    for cell_i in range(S):
        for cell_j in range(S):
            box[N,cell_i,cell_j,:4] = IoU.relative2absolute_pred(box, cell_i, cell_j) #(N,4)
            iou = IoU.intersection_over_union(box[N,cell_i,cell_j,:4], box_max_confidence[:,:4])
            iou_bool = iou >= iou_threshold
                        
            ### iou to shape (N,4)
            iou_bool = iou_bool.unsqueeze(1).repeat((1, box.shape[-1]))
            box[:,cell_i, cell_j] = box[:,cell_i, cell_j].masked_fill(iou_bool, 0)

            mask = box[:,cell_i, cell_j] < 0
            box[:,cell_i, cell_j] = 0 
            box[:,cell_i, cell_j].masked_fill(mask, 0)

    # 5) TODO
    # print("DEBUG : ", box_max_confidence.shape, box_max_confidence, sep='\n')
    box[N, m[:,0], m[:,1], :4] = box_max_confidence[:,:4]
    return box, m