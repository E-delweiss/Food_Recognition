import torch

import IoU
import utils

def non_max_suppression(prediction, prob_threshold, iou_threshold):
    """
    Use confidence number and class probabilities such as pc = pc * P(C)
    to keep only predicted boxes with the highest confidence number.
    Convert boxes into absolute coordinates and compute IoU.
    prediction is (1,7,7,18)
    TODO
    """
    S = 7
    N = range(len(prediction))

    ### Create a mask regarding prob_threshold for each bbox in each cell
    mask_box1 = prediction[...,4].lt(prob_threshold).unsqueeze(3)
    mask_box2 = prediction[...,9].lt(prob_threshold).unsqueeze(3)
    mask_box1 = mask_box1.repeat(1,1,1,5)
    mask_box2 = mask_box2.repeat(1,1,1,5)
    mask_box = torch.concat((mask_box1, mask_box2), dim=-1) # -> (1,7,7,10)

    ### Save labels
    prediction_label = prediction[...,10:]

    for i in range(S):
        for j in range(S):
                prediction[:,i,j,0:5] = IoU.relative2absolute(prediction[...,:5], N, i, j)
                prediction[:,i,j,5:10] = IoU.relative2absolute(prediction[...,5:10], N, i, j)

    ### Zeroed all box for which pc < prob_threshold
    prediction_masked = torch.masked_fill(prediction[...,:10], mask_box, 0)
    cell_i, cell_j, _ = prediction_masked.nonzero().permute(1,0)

    ### Retrieve boxes and label with pc>0
    prediction_label = torch.argmax(torch.softmax(prediction_label[:,cell_i, cell_j], dim=-1), dim=-1).unsqueeze(-1)
    prediction = torch.concat((prediction_masked[:,cell_i,cell_j,:5], prediction_masked[:,cell_i,cell_j,5:10]), dim=1)
    prediction = torch.concat((prediction, prediction_label.repeat(1,2)), dim=-1)
    prediction = prediction.unique(dim=1)

    ### NMS
    prediction_list = prediction.tolist()
    prediction_list = sorted(prediction_list, key=lambda x: x[4])
    


    return bbox_nms