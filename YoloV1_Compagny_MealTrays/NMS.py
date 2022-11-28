from icecream import ic
import torch

import IoU
import utils

def non_max_suppression(prediction:torch.Tensor, prob_threshold:float, iou_threshold:float)->list:
    """
    Apply non max suppression algorithm.

    Args:
        prediction (torch.Tensor of shape (N,S,S,5+C*B))
            Predicted tensor containing 2 bounding boxes per grid cell.
        prob_threshold (float)  
            Threshold set to discard all predicted bounding box with smaller probability than
            a selected bounding box.
        iou_threshold (float)
            If the intersection over union between two bboxes is larger than this threshold, then
            those bboxes are not predicted for the same object.

    Returns:
        bboxes_nms (list)
            List of all bounding box for the predicted tensor after NMS.
            Contains infos such as [box1[x,y,w,h,c,label], box2[...],...]
    """
    prediction_temp = torch.concat((prediction[...,:5], prediction[...,10:]), dim=-1)
    prediction_abs_box1 = IoU.relative2absolute(prediction_temp)
    prediction_abs_box2 = IoU.relative2absolute(prediction[...,5:])
    
    # [img1[box1[x,y,w,h,c,label], ...], img2[box1[...]], ...]
    list_box1 = utils.tensor2boxlist(prediction_abs_box1)
    list_box2 = utils.tensor2boxlist(prediction_abs_box2)
    list_all_boxes = list_box1 + list_box2

    bboxes = [box for box in list_all_boxes if box[4] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)
    bboxes_nms = []
    
    while bboxes:
        box_candidate = bboxes.pop(0)
        bboxes = [box for box in bboxes if box[5] != box_candidate[5] 
            or IoU.intersection_over_union(
                torch.tensor(box_candidate[:4]).unsqueeze(0), 
                torch.tensor(box[:4]).unsqueeze(0)) < iou_threshold
        ]

        bboxes_nms.append(box_candidate)

    return bboxes_nms