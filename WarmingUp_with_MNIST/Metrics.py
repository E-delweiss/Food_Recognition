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



def test():
    from MNIST_dataset import get_training_dataset
    
    S = 6
    BATCH_SIZE=16
    labels_pred = torch.rand(BATCH_SIZE, S, S, 10)

    training_dataset, _ = get_training_dataset(BATCH_SIZE)
    for img, boxes_true, labels_true in training_dataset:
        break
    
    acc = class_acc(boxes_true, labels_true, labels_pred)
    print(acc)

if __name__ == '__main__':
    test()