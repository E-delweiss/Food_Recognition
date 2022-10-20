import torch
import utils
import IoU

def class_acc(target:torch.Tensor, prediction:torch.Tensor, B:int=2)->float:
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
    softmax_pred_classes = torch.softmax(prediction[N, cells_i, cells_j, B*5:], dim=1)
    labels_pred = torch.argmax(softmax_pred_classes, dim=-1)

    ### Get true labels
    labels_true = torch.argmax(target[N, cells_i, cells_j, 5:], dim=-1)
    
    ### Mean of the right predictions where there should be an object
    acc = (1/len(N)) * torch.sum(labels_true == labels_pred)
    return acc.item()


def MSE(target:torch.Tensor, prediction:torch.Tensor, S=7, B=2)->float:
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
    iou_box = []
    target_box_abs = IoU.relative2absolute(target[:,:,:,:5], N, cells_i, cells_j) # -> (N,4)
    for b in range(B):
        box_k = 5*b
        prediction_box_abs = IoU.relative2absolute(prediction[:,:,:, box_k : 5+box_k], N, cells_i, cells_j) # -> (N,4)
        iou = IoU.intersection_over_union(target_box_abs, prediction_box_abs) # -> (N,1)
        iou_box.append(iou) # -> [iou_box1:(N), iou_box2:(N)]                    
    
    ### TODO comment
    box_mask = torch.lt(iou_box[0], iou_box[1]).to(torch.int64)
    idx = 5*box_mask #if 0 -> box1 infos, if 5 -> box2 infos

    ### bbox coordinates relating to the box with the largest IoU
    ### note : python doesn't like smth like a[N,i,j, arr1:arr2]
    x_hat = prediction[N, cells_i, cells_j, idx]                
    y_hat = prediction[N, cells_i, cells_j, idx+1]                
    w_hat = prediction[N, cells_i, cells_j, idx+2]
    h_hat = prediction[N, cells_i, cells_j, idx+3]
    
    xywh_hat = torch.stack((x_hat, y_hat, w_hat, h_hat), dim=-1)
    xywh = target[N, cells_i, cells_j, :4]

    MSE_score = torch.pow(xywh - xywh_hat,2)
    
    ### Mean of the box info MSEs
    MSE_score = (1/len(N)) * torch.sum(MSE_score)
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
    ### Retrieve indices with object
    N, cells_i, cells_j = utils.get_cells_with_object(target)
    
    ### Compute the losses for all images in the batch
    iou_box = []
    target_box_abs = IoU.relative2absolute(target[:,:,:,:5], N, cells_i, cells_j) # -> (N,4)
    for b in range(B):
        box_k = 5*b
        prediction_box_abs = IoU.relative2absolute(prediction[:,:,:, box_k : 5+box_k], N, cells_i, cells_j) # -> (N,4)
        iou = IoU.intersection_over_union(target_box_abs, prediction_box_abs) # -> (N,1)
        iou_box.append(iou) # -> [iou_box1:(N), iou_box:(N)]                    
    
    ### TODO comment
    iou_mask = torch.lt(iou_box[0], iou_box[1]).to(torch.int64)
    idx = 5*iou_mask #if 0 -> box1 infos, if 5 -> box2 infos

    ### confident score related to the box with the largest IoU
    prediction_confident_score = prediction[N, cells_i, cells_j, idx+4]                
    target_confident_score = target[N, cells_i, cells_j, 4]

    # iou = intersection_over_union(target[:,i,j], prediction[:,i,j]).to(device)
    mse_confidence_score = torch.pow(target_confident_score - prediction_confident_score, 2) #* iou,  2)

    ### Mean of the confidence scores
    mse_confidence_score = (1/len(N)) * torch.sum(mse_confidence_score)
    return mse_confidence_score.item()