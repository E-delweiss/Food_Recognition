import torch

def validation_loop(model, validation_dataset, S=7, device=torch.device("cpu")):
    """
    TODO
    Execute validation loop

    Args:
        model (nn.Module)
            Yolo model.
        validation_dataset (Dataset) 
            Validation dataloader
        S (int, optional)
            Grid size. Defaults to 7.
        device (torch.device, optional)
            Running device. Defaults to cpu

    Returns:
        img (torch.Tensor of shape (N,3,448,448))
            Images from validation dataset
        target (torch.Tensor of shape (N,S,S,(4+1)+C) -> (N,7,7,13))
            Groundtruth box informations
        prediction (torch.Tensor of shape (N,S,S,B*(4+1)+C) -> (N,7,7,18))
            Predicted box informations  
    """
    print("|")
    print("| Validation...")
    for (img, target) in validation_dataset:
        img, target = img.to(device), target.to(device)
        
        with torch.no_grad():
            ### prediction
            prediction = model(img)

    return img, target, prediction



