import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

import torch

from smallnet import netMNIST
from MNIST_dataset import get_validation_dataset
import utils
import NMS, IoU, mAP

S=6
B=2
C=10
frame_size=140
PROB_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
device = torch.device("mps")


model = netMNIST(load_weights=True, sizeHW=frame_size, S=S, B=B, C=C).to(torch.device("mps"))

validation_dataset = get_validation_dataset(BATCH_SIZE=512)

model.eval()
mAP_list1 = []
mAP_list2 = []
train_idx = 0

for IOU_THRESHOLD in np.arange(0.5, 1.0, 0.05):
    for batch, (img, target) in enumerate(validation_dataset):
        img, target = img.to(device), target.to(device)
        with torch.no_grad():
            ### prediction
            predictions = model(img)

        all_pred_boxes = []
        all_true_boxes = []
        for idx in range(len(target)):
            true_bboxes = IoU.relative2absolute(target[idx].unsqueeze(0), frame_size)
            true_bboxes = utils.tensor2boxlist(true_bboxes)

            nms_box_val = NMS.non_max_suppression(predictions[idx].unsqueeze(0), frame_size, PROB_THRESHOLD, IOU_THRESHOLD)

            for nms_box in nms_box_val:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes:
                # many will get converted to 0 pred
                if box[4] > PROB_THRESHOLD:
                    all_true_boxes.append([train_idx] + box)
            
            train_idx += 1

        current_map = mAP.mean_average_precision(all_true_boxes, all_pred_boxes, IOU_THRESHOLD, S, C)
        mAP_list1.append(current_map)
        print(f"BATCH {batch+1}/{len(validation_dataset)}, IOU_THRESHOLD={IOU_THRESHOLD} : mAP = {current_map:.3f}")
    
    mAP_list2.append(np.mean(mAP_list1))

with open('pickle_mAP_MNIST.pkl', 'wb') as pkl:
    pickle.dump(mAP_list2, pkl)


