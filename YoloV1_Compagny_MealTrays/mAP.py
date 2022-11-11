import sys
from collections import Counter

import torch

import IoU
import utils


def mean_average_precision(target, prediction, iou_threshold):
    """
    TODO
    """
    C = 8
    S=7
    B=2
    average_precisions = []
    epsilon = 1e-6

    for c in range(C):
        detections = []
        ground_truths = []

        for detection in prediction:
            if detection[-1] == c:
                detections.append(detection)

        for true_box in target:
            if true_box[-1] == c:
                ground_truths.append(true_box)
        
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[-1], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # print(len(detections))

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = IoU.intersection_over_union(
                    torch.tensor(detection[1:5]).unsqueeze(0),
                    torch.tensor(gt[1:5]).unsqueeze(0),
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        from icecream import ic
        ic(TP)
        ic(FP)

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)



if __name__ == '__main__':
    import torch

    from mealtrays_dataset import get_validation_dataset
    from resnet101 import resnet
    from validation import validation_loop

    # model = darknet(True, in_channels=3, S=7, C=8, B=2)
    # model = resnet(True, in_channels=3, S=7, C=8, B=2)
    # model.eval()
    # validation_dataset = get_validation_dataset(32, isNormalize=False)
    # n_img, n_target, n_prediction = validation_loop(model, validation_dataset, ONE_BATCH=True)
    # print(mean_average_precision(n_target, n_prediction, 0.5))
