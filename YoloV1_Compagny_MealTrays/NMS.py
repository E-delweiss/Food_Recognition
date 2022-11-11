from icecream import ic
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
    # ic(torch.concat((prediction[...,:5], prediction[...,10:]), dim=-1).shape)
    prediction_temp = torch.concat((prediction[...,:5], prediction[...,10:]), dim=-1)
    prediction_abs_box1 = IoU.relative2absolute(prediction_temp)
    prediction_abs_box2 = IoU.relative2absolute(prediction[...,5:])
    
    # [img1[box1[x,y,w,h,c,label], ...], img2[box1[...]], ...]
    list_box1 = utils.tensor2boxlist(prediction_abs_box1)
    list_box2 = utils.tensor2boxlist(prediction_abs_box2)
    list_all_boxes = list_box1 + list_box2

    # ic(list_box1)

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