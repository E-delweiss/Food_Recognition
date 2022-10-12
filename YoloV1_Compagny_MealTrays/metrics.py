import torch
import utils


def class_acc(target:torch.Tensor, prediction:torch.Tensor, B=2)->float:
    """
    Compute class accuracy using cells WITH object.

    Args: 
        target : torch.Tensor of shape (N,S,S,4+1+C) -> (N,7,7,13)
            Groundtruth tensor
        prediction : torch.Tensor of shape (N,S,S,B*(4+1)+C) -> (N,7,7,18)
            Predicted tensor

    Returns:
        acc : float
            Class accuracy between 0 and 1.
    """
    ### Current batch size
    BATCH_SIZE = len(target)

    ### Retrieve indices with object
    N, cells_i, cells_j = utils.get_cells_with_object(target)

    ### Applying softmax to get label probabilities only in cells with object
    softmax_pred_classes = torch.softmax(prediction[N, cells_i, cells_j, B*(4+1):], dim=1)
    labels_pred = torch.argmax(softmax_pred_classes, dim=-1)

    ### Get true labels
    labels_true = torch.argmax(target[N, cells_i, cells_j, (4+1):], dim=-1)

    ### Mean of the right predictions where there should be an object
    acc = (1/BATCH_SIZE) * torch.sum(labels_true == labels_pred)
    return acc.item()


def MSE(target:torch.Tensor, prediction:torch.Tensor, B=2)->float:
    """
    Mean Square Error along bbox coordinates and sizes in the cells containing an object

    Args: 
        target : torch.Tensor of shape (N,S,S,4+1+C) -> (N,7,7,13)
            Groundtruth tensor
        prediction : torch.Tensor of shape (N,S,S,B*(4+1)+C) -> (N,7,7,18)
            Predicted tensor
    
    Returns:
        MSE_score : float
            MSE value between 0 and 1
    """
    ### Current batch size
    BATCH_SIZE = len(target)

    ### Retrieve indices with object
    N, cells_i, cells_j = utils.get_cells_with_object(target)

    ### Get box info only in cells with object
    target = target[N, cells_i, cells_j, 0:4]

    MSE_score = 0
    for b in range(B):
        box_k = b*(4+1)
        prediction_box_k = prediction[N, cells_i, cells_j, 0+box_k :4+box_k]
        MSE_score += torch.pow(target - prediction_box_k,2)
    
    ### Mean of the box info MSEs
    MSE_score = (1/BATCH_SIZE) * torch.sum(MSE_score)
    return MSE_score.item()


def MSE_confidenceScore(target:torch.Tensor, prediction:torch.Tensor, S:int=7, B:int=2)->float:
    """
    _summary_

    Args:
        target : torch.Tensor of shape (N,S,S,4+1+C) -> (N,7,7,13)
            Groundtruth tensor
        prediction : torch.Tensor of shape (N,S,S,B*(4+1)+C) -> (N,7,7,18)
            Predicted tensor
        S : int
            Grid size
        B : int
            Predicted box number

    Returns:
        mse_confidence_score : float
    """
    ### Current batch size
    BATCH_SIZE = len(target)

    mse_confidence_score = 0
    for i in range(S):
        for j in range(S):
            for b in range(B):
                box_k = b*(4+1)
                # iou = intersection_over_union(target[:,i,j], prediction[:,i,j]).to(device)
                mse_confidence_score += torch.pow(target[:,i,j,4] - prediction[:,i,j,4+box_k], 2) #* iou,  2)

    ### Mean of the confidence scores
    mse_confidence_score = (1/BATCH_SIZE) * torch.sum(mse_confidence_score)
    return mse_confidence_score.item()
