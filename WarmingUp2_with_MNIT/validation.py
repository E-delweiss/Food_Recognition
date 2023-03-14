import torch

def validation_loop(model, validation_dataset, S=6, device=torch.device("cpu")):
    """
    Execute validation loop

    Args:
        model (nn.Module)
            Yolo model.
        validation_dataset (Dataset) 
            Validation dataset from the MNIST_dataset.py module.
        S (int, optional)
            Grid size. Defaults to 7.
        device (torch.device, optional)
            Running device. Defaults to cpu

    Returns:
        img (torch.Tensor of shape (N,1,75,75))
            Images from validation dataset
        bbox_true (torch.Tensor of shape (N,S,S,5))
            Groundtruth bounding boxes
        bbox_pred (torch.Tensor of shape (N,S,S,5))
            Predicted bounding boxes
        labels (torch.Tensor of shape (N,S,S,10))
            Groundtruth labels
        labels_pred (torch.Tensor of shape (N,S,S,10))
            Predicted labels
    """
    model.eval()
    print("|")
    print("| Validation...")
    for (img, bbox_true, labels) in validation_dataset:
        img, bbox_true, labels  = img.to(device), bbox_true.to(device), labels.to(device)
        
        with torch.no_grad():
            ### prediction (N,S,S,5) & (N,S,S,10)
            bbox_pred, labels_pred = model(img)
        break
    return img, bbox_true, bbox_pred, labels, labels_pred



