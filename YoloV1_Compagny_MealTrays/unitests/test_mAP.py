import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

current_folder = os.path.dirname(locals().get("__file__"))
parent_folder = Path(current_folder).parent
sys.path.append(str(parent_folder))

from icecream import ic

import IoU
import mAP
import NMS
import utils
from mealtrays_dataset import get_validation_dataset
from resnet101 import resnet
from validation import validation_loop


class TestmAP(unittest.TestCase):
    def __init__(self, TestmAP) -> None:
        super().__init__(TestmAP)
        self.size = 448
        self.S = 7
        self.B = 2
        self.C = 8
        self.channel_img = 3
        self.BATCH_SIZE = 16
        self.val_loader = get_validation_dataset(self.BATCH_SIZE, isAugment=False)
        self.model = resnet(pretrained=True, in_channels=3, S=7, B=2, C=8)
        self.model.eval()

    def test_mAP(self):      
        prob_threshold = 0.4
        iou_threshold = 0.5
        
        train_idx = 0
        _, target_val, prediction_val = validation_loop(self.model, self.val_loader, self.S, ONE_BATCH=True)
        
        all_pred_boxes = []
        all_true_boxes = []
        for idx in range(len(target_val)):
            true_bboxes = IoU.relative2absolute(target_val[idx].unsqueeze(0))
            true_bboxes = utils.tensor2boxlist(true_bboxes)

            nms_box_val = NMS.non_max_suppression(prediction_val[idx].unsqueeze(0), prob_threshold, iou_threshold)

            for nms_box in nms_box_val:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes:
                # many will get converted to 0 pred
                if box[4] > prob_threshold:
                    all_true_boxes.append([train_idx] + box)
            
            train_idx += 1

        MAP = mAP.mean_average_precision(all_true_boxes, all_pred_boxes, iou_threshold)
        print(MAP)

if __name__ == "__main__":
    unittest.main()
        
