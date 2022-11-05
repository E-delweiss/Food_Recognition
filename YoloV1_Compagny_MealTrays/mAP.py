import torch

from collections import Counter

import IoU
import utils
import sys

def mean_average_precision(target, prediction, iou_threshold):
    """
    TODO
    """
    C = 8
    S=7
    B=2
    BATCH_SIZE = 32

    target_labels = torch.argmax(target[...,5:],dim=-1)
    target = torch.concat((target[...,:5], target_labels.unsqueeze(-1)), dim=-1)
    prediction = torch.concat((
        prediction[...,:10], torch.argmax(torch.softmax(prediction[...,10:],dim=-1),dim=-1).unsqueeze(-1)
        ),dim=-1)

    # list storing all AP for respective classes
    average_precisions = []

    smoothing_factor = 1e-6


    N, cells_i, cells_j = utils.get_cells_with_object(target)
    # for i in range(S):
    #     for j in range(S):
    prediction[N,cells_i,cells_j,:5] = IoU.relative2absolute(prediction[...,:5], N,cells_i,cells_j)
    prediction[N,cells_i,cells_j,5:10] = IoU.relative2absolute(prediction[...,5:10], N,cells_i,cells_j)
    target[N,cells_i,cells_j,:5] = IoU.relative2absolute(target[...,:5], N,cells_i,cells_j)


    # print(target[4])

    # sys.exit()

    for c in range(C):
        detections = []
        ground_truths = []

        for k in range(BATCH_SIZE):
            for i in cells_i:
                for j in cells_j:
                    if prediction[k,i,j,10] == c:
                        detections.append([k]+prediction[k,i,j,:5].tolist())
                        detections.append([k]+prediction[k,i,j,5:10].tolist())
                    
                    if target[k,i,j,5] == c :
                        ground_truths.append([k]+target[k,i,j,:5].tolist())
        
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[5], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        print(len(detections))

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            print("ok2")
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

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + smoothing_factor)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + smoothing_factor))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)



if __name__ == '__main__':
    from resnet101 import resnet
    import torch
    from mealtrays_dataset import get_validation_dataset
    from validation import validation_loop

    # model = darknet(True, in_channels=3, S=7, C=8, B=2)
    model = resnet(True, in_channels=3, S=7, C=8, B=2)
    model.eval()
    validation_dataset = get_validation_dataset(32, isNormalize=False)
    n_img, n_target, n_prediction = validation_loop(model, validation_dataset, ONE_BATCH=True)

    print(mean_average_precision(n_target, n_prediction, 0.5))
