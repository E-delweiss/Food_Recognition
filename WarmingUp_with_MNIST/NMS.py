from IoU import relative2absolute, intersection_over_union

#### TODO
def non_max_suppression(box_pred, label_pred, iou_threshold):
    """
    - Get the highest pc number
    - Discard all other bbox with a HIGH IOU (bc if the iou is low, it means this pc number doesn't stand for the current object)

    Args :
        bbox_pred : torch.Tensor of shape (N, S, S, 5)
            Predicted bounding boxes with x, y, w, h and confident pc number
        labels_pred: torch.Tensor of shape (N, S, S, 10)
            Predicted label
        iou_threshold : float
            Every bbox iou bigger than iou_threshold will be discard
        
    """
    def find_indices_max(tens):
        """
        input tens : torch.Tensor of shape (N, S, S)
        output batch_indices : torch.Tensor of shape (N,2)
        """
        S = tens.shape[1]
        N = tens.shape[0]

        # Reshape to (N, S*S)
        tens_reshape = tens.view(N, S*S)
        indices = torch.argmax(tens_reshape, dim=1)

        col_indices = (indices / S).to(torch.int32)
        row_indices = indices % S

        batch_indices = torch.stack((col_indices, row_indices)).T
        return batch_indices
        #########################################################
    bbox = torch.clone(box_pred)
    labels = torch.clone(label_pred)
    N = len(bbox)
    S = bbox.shape[1]

    labels_prob = torch.softmax(labels, dim=-1)

    bbox[:,:,:,4] = torch.mul(bbox[:,:,:,4], torch.max(labels_prob, dim=-1)[0])

    ### 1) finding indices i,j of the max confidence number of each image in the batch
    m = find_indices_max(bbox[:,:,:,4])

    ### Getting bboxes with the highest pc number for each image
    ### Shape : (N, 4)
    bbox_max_confidence = bbox[range(N), m[:,0], m[:,1]]


    ### Removing bboxes with the highest pc numbers
    bbox[range(N), m[:,0], m[:,1]] = torch.Tensor([0])

    for cell_i in range(S):
        for cell_j in range(S):
            iou = intersection_over_union(bbox[:,cell_i, cell_j], bbox_max_confidence)
            iou_bool = iou >= iou_threshold
            
            ### iou to shape (N,4)
            iou_bool = iou_bool.unsqueeze(1).repeat((1, bbox.shape[-1]))
            bbox[:,cell_i, cell_j] = bbox[:,cell_i, cell_j].masked_fill(iou_bool, 0)

            mask = bbox[:,cell_i, cell_j] < 0
            bbox[:,cell_i, cell_j] = 0 # bbox[:,cell_i, cell_j].masked_fill(mask, 0)
            
            #### stacker bbox[:,cell_i, cell_j] > 0 ET bbox_max_confidence

    bbox[range(N), m[:,0], m[:,1]] = bbox_max_confidence
    return bbox