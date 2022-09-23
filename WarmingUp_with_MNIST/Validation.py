import torch
from tqdm import tqdm

from Metrics import MSE, class_acc
from IoU import intersection_over_union

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
        mse_score (float)
            MSE score between true and pred boxes in cells containing object.
        mse_confidence_score (float)
            Mean of the all grid confidence scores.
        acc (float)
            Class accuracy in cells containing object.
    """
    model.eval()
    print("|")
    print("| Validation...")
    for (img, bbox_true, labels) in validation_dataset:
        img, bbox_true, labels  = img.to(device), bbox_true.to(device), labels.to(device)
        
        with torch.no_grad():
            ### prediction (N,S,S,5) & (N,S,S,10)
            bbox_pred, labels_pred = model(img)

            ### MSE score
            mse_score = MSE(bbox_true, bbox_pred)
            
            ### confidence score 
            mse_confidence_score = torch.zeros(len(img)).to(device)
            for i in range(S):
                for j in range(S):
                    # iou = intersection_over_union(bbox_true[:,i,j], bbox_pred[:,i,j]).to(device)
                    mse_confidence_score += torch.pow(bbox_true[:,i,j,-1] - bbox_pred[:,i,j,-1], 2) #* iou,  2)
            
            mse_confidence_score = (1/(len(img))) * torch.sum(mse_confidence_score)

            ### Accuracy
            acc = class_acc(bbox_true, labels, labels_pred)

            #####################################################################
            print(f"| MSE validation box loss : {mse_score:.5f}")
            print(f"| MSE validation confidence score : {mse_confidence_score.item():.5f}")
            print(f"| Validation class acc : {acc*100:.2f}%")
            print("\n")

            return mse_score, mse_confidence_score.item(), acc


