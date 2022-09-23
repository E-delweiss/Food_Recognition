import torch


def class_acc(box_true:torch.Tensor, labels_true:torch.Tensor, labels_pred:torch.Tensor)->float:
    """
    Compute class accuracy using cells WITH object.

    Args: 
        box_true : torch.Tensor of shape (N, S, S, 5)
            Groundtruth box tensor
        labels_true : torch.Tensor of shape (N, 10)
            Groundtruth one hot encoded labels
        labels_pred : torch.Tensor of shape (N, S, S, 10)
            Predicted labels

    Returns:
        acc : float
            Class accuracy between 0 and 1.
    """

    ### Current batch size
    BATCH_SIZE = len(box_true)

    ### Retrieve indices with object (TBM for box > 1)
    cells_with_obj = box_true.nonzero()[::5]
    N, cells_i, cells_j, _ = cells_with_obj.permute(1,0)

    ### Applying softmax to get probability
    softmax_pred_classes = torch.softmax(labels_pred[N, cells_i, cells_j], dim=1)

    ### Mean of the right predictions where there should be an object
    acc = (1/BATCH_SIZE) * torch.sum(torch.argmax(labels_true, dim=1) == torch.argmax(softmax_pred_classes, dim=1))
    
    return acc.item()


def MSE(box_true:torch.Tensor, box_pred:torch.Tensor)->float:
    """
    Mean Square Error along bbox coordinates and sizes in the cells containing an object

    Args:
        box_true : torch.Tensor of shape (N, S, S, 5)
            Groundtruth box tensor
        box_pred : torch.Tensor of shape (N, S, S, 5)
            Predicted box tensor
    
    Returns:
        MSE_score : float
            MSE value between 0 and 1
    """
    BATCH_SIZE = len(box_true)

    cells_with_obj = box_true.nonzero()[::5]
    N, cells_i, cells_j, _ = cells_with_obj.permute(1,0)

    ### (N,S,S,5) -> (N,4)
    box_true = box_true[N, cells_i, cells_j, 0:4]
    box_pred = box_pred[N, cells_i, cells_j, 0:4]

    MSE_score = torch.pow(box_true - box_pred,2)
    MSE_score = (1/BATCH_SIZE) * torch.sum(MSE_score)

    return MSE_score.item()


def MSE_confidenceScore(bbox_true:torch.Tensor, bbox_pred:torch.Tensor, S=6, device:torch.device=torch.device('cpu'))->float:
    """
    _summary_

    Args:
        bbox_true (torch.Tensor of shape (N,S,S,5))
            Groundtruth box tensor.
        bbox_pred (torch.Tensor of shape (N,S,S,5))
            Predicted box tensor.
        device (torch.device, optional): 
            Defaults to CPU.

    Returns:
        mse_confidence_score (float)
    """
    
    mse_confidence_score = torch.zeros(len(bbox_true)).to(device)
    for i in range(S):
        for j in range(S):
            # iou = intersection_over_union(bbox_true[:,i,j], bbox_pred[:,i,j]).to(device)
            mse_confidence_score += torch.pow(bbox_true[:,i,j,-1] - bbox_pred[:,i,j,-1], 2) #* iou,  2)
            
    mse_confidence_score = (1/(len(bbox_true))) * torch.sum(mse_confidence_score).item()
    return mse_confidence_score




def test():
    from MNIST_dataset import get_training_dataset
    
    S = 6
    BATCH_SIZE=16
    label_pred = torch.rand(BATCH_SIZE, S, S, 10)

    training_dataset, _ = get_training_dataset(BATCH_SIZE)
    for img, box_true, label_true in training_dataset:
        break
    
    acc = class_acc(box_true, label_true, label_pred)
    print(acc)

    box_pred = torch.rand(BATCH_SIZE, S, S, 5)
    mse_score = MSE(box_true, box_pred)
    print(mse_score)

if __name__ == '__main__':
    test()