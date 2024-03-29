import torch
import utils

def class_acc(target:torch.Tensor, prediction:torch.Tensor)->float:
    """
    Compute class accuracy using cells WITH object.

    Args: 
        target : torch.Tensor of shape (N,S,S,5+C)
            Groundtruth tensor
        prediction : torch.Tensor of shape (N,S,S,B*5+C)
            Predicted tensor

    Returns:
        acc : float
            Class accuracy: mean of the true predictions (TP+TN)/nb_pred  
        hard_acc : float
            Hard accuracy: score to zero an image if 1 object is not correctly classify
    """
    ### Retrieve indices with object
    N, cells_i, cells_j = utils.get_cells_with_object(target)

    ### Applying softmax to get label probabilities only in cells with object
    softmax_pred_classes = torch.softmax(prediction[N, cells_i, cells_j, 10:], dim=-1)
    labels_pred = torch.argmax(softmax_pred_classes, dim=-1)

    ### Get true labels
    labels_true = torch.argmax(target[N, cells_i, cells_j, 5:], dim=-1)
    
    ### Mean of the true predictions where there should be an object
    acc = (1/len(N)) * torch.sum(labels_true == labels_pred).item()

    ### Compute hard accuracy : 0 if an object is not correctly classify
    N, cells_i, cells_j = N.to("cpu"), cells_i.to("cpu"), cells_j.to("cpu")
    N_tab = torch.stack((torch.unique(N), torch.bincount(N), torch.cumsum(torch.bincount(N), dim=-1)), dim=-1)

    start = N_tab.permute(1,0)[2] - N_tab.permute(1,0)[1]
    end = start + N_tab.permute(1,0)[1]

    count = 0
    for s, e in zip(start, end):
        if torch.all(torch.eq(labels_true[s:e], labels_pred[s:e])):
            count += 1
    
    ### Mean of the right predictions
    hard_acc = (1/len(torch.unique(N))) * count

    return acc, hard_acc