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
    softmax_pred_classes = torch.softmax(prediction[N, cells_i, cells_j, 5:], dim=1)
    labels_pred = torch.argmax(softmax_pred_classes, dim=-1)

    ### Get true labels
    labels_true = torch.argmax(target[N, cells_i, cells_j, 5:], dim=-1)
    
    ### Mean of the right predictions where there should be an object
    acc = (1/len(N)) * torch.sum(labels_true == labels_pred)
    return acc.item()


def MSE(target:torch.Tensor, prediction:torch.Tensor)->float:
    """
    Mean Square Error along bbox coordinates and sizes in the cells containing an object

    Args: 
        target : torch.Tensor of shape (N,S,S,4+1+C) -> (N,7,7,13)
            Groundtruth tensor
        prediction : torch.Tensor of shape (N,S,S,B*(4+1)+C) -> (N,7,7,18)
            Predicted tensor
    
    Returns:
        MSE_score : float
            MSE value
    """
    ### Retrieve indices with object
    N, cells_i, cells_j = utils.get_cells_with_object(target)
    
    ### Compute the losses for all images in the batch
    target = target[N, cells_i, cells_j, 0:4] # -> (N,4)
    prediction = prediction[N, cells_i, cells_j, 0:4]

    MSE_score = torch.pow(target - prediction,2)
    
    ### Mean of the box info MSEs
    MSE_score = (1/len(N)) * torch.sum(MSE_score)
    return MSE_score.item()
    

def MSE_confidenceScore(target:torch.Tensor, prediction:torch.Tensor)->float:
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
    ### Retrieve indices with object
    N, cells_i, cells_j = utils.get_cells_with_object(target)
    
    ### Compute the losses for all images in the batch
    target_confident_score = target[N, cells_i, cells_j, 4] # -> (N,4)
    prediction_confident_score = prediction[N, cells_i, cells_j, 4]

    # iou = intersection_over_union(target[:,i,j], prediction[:,i,j]).to(device)
    mse_confidence_score = torch.pow(target_confident_score - prediction_confident_score, 2) #* iou,  2)

    ### Mean of the confidence scores
    mse_confidence_score = (1/len(N)) * torch.sum(mse_confidence_score)
    return mse_confidence_score.item()