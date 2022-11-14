import torch
import utils
import IoU

def class_acc(target:torch.Tensor, prediction:torch.Tensor)->float:
    """
    Compute class accuracy using cells WITH object.

    Args: 
        target : torch.Tensor of shape (N,S,S,4+1+C) -> (N,7,7,13)
            Groundtruth tensor
        prediction : torch.Tensor of shape (N,S,S,B*(4+1)+C) -> (N,7,7,18)
            Predicted tensor
        B : int
            Number of predicted bounding box

    Returns:
        acc : float
            Class accuracy between 0 and 1.
    """
    ### Retrieve indices with object
    N, cells_i, cells_j = utils.get_cells_with_object(target)

    ### Applying softmax to get label probabilities only in cells with object
    softmax_pred_classes = torch.softmax(prediction[N, cells_i, cells_j, 10:], dim=-1)
    labels_pred = torch.argmax(softmax_pred_classes, dim=-1)

    ### Get true labels
    labels_true = torch.argmax(target[N, cells_i, cells_j, 5:], dim=-1)
    
    ### Mean of the right predictions where there should be an object
    acc = (1/len(N)) * torch.sum(labels_true == labels_pred)

    ### Compute hard accuracy : 0 if one object is not correctly classify
    N_tab = torch.stack((torch.unique(N), torch.bincount(N), torch.cumsum(torch.bincount(N), dim=-1)), dim=-1)

    start = N_tab.permute(1,0)[2] - N_tab.permute(1,0)[1]
    end = start + N_tab.permute(1,0)[1]

    count = 0
    for s, e in zip(start, end):
        if torch.all(torch.eq(labels_true[s:e], labels_pred[s:e])):
            count += 1
    
    ### Mean of the right predictions
    hard_acc = (1/len(torch.unique(N))) * count

    return acc.item(), hard_acc