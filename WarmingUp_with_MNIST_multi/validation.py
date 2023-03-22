import torch

def validation_loop(model, validation_dataset, S=6, device=torch.device("cpu"), ONE_BATCH=True):
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
    for (img, target) in validation_dataset:
        img, target = img.to(device), target.to(device)
        
        with torch.no_grad():
            ### prediction
            prediction = model(img)
            
            if ONE_BATCH is True:
                break

    return img, target, prediction



