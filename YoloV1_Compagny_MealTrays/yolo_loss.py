import torch
import IoU
import datetime

class YoloLoss(torch.nn.Module):
    def __init__(self, lambd_coord:int, lambd_noobj:float, device:torch.device, S:int=7, B:int=2):
        super(YoloLoss, self).__init__()
        self.LAMBD_COORD = lambd_coord
        self.LAMBD_NOOBJ = lambd_noobj
        self.S = S
        self.B = B
        self.device = device
        self.MSE = torch.nn.MSELoss(reduction='sum')
        self.EPS = 1e-6

    def _MSELoss(self, pred:torch.Tensor, target:torch.Tensor, isObject:torch.Tensor, isRMSE:bool=False)->torch.Tensor:
        """
        Mask input tensors regarding if there should be an object in the cell or not.
        Compute MSE loss or RMSE loss.

        Args:
            pred: torch.Tensor of shape (N,S,S,_)
                Predicted inputs of the loss
            target: torch.Tensor of shape (N,S,S,_)
                Groundtruth inputs of the loss
            isObject: torch.Tensor of shape (N,S,S,1)
                Boolean tensor used to mask values
            isRMSE bool, optional: 
                Active RMSE loss. Defaults to False.

        Returns:
            torch.Tensor: _description_
        """
        size = len(target)
        expand_shape = pred.shape[-1]

        isObject = isObject.repeat(1,1,1, expand_shape)
        pred = torch.masked_fill(pred, isObject, 0)
        target = torch.masked_fill(target, isObject, 0)

        output = self.MSE(target, pred)
        if isRMSE:
            output = torch.sqrt(output + self.EPS)
        return output/size


    def forward(self, prediction:torch.Tensor, target:torch.Tensor)->tuple[dict, torch.Tensor]:
        """
        Grid fowrard pass.

        Args:
            prediction : torch.Tensor of shape (N, S, S, B*5+C)
                Predicted batch outputs containing 2 box infos (xcr_rcell, ycr_rcell, wr, hr, c)
                and a one-hot encoded class for each grid cell.

            target : torch.Tensor of shape (N, S, S, 5+C)
                Groundtrouth batch outputs containing (xcr_rcell, ycr_rcell, wr, hr, c)
                and one-hot encoded class for each grid cell.

        Return:
            losses : dict
                Dictionnary of each loss used to compute the global loss.
            loss : float
                The loss value of the batch.
        """
        BATCH_SIZE = len(target)

        ### Initialization of the losses
        losses_list = ['loss_xy', 'loss_wh', 'loss_conf_obj', 'loss_conf_noobj', 'loss_class']
        losses = {key : torch.zeros(BATCH_SIZE).to(self.device) for key in losses_list}
        
        ### Get absolute coordinates: (N,S,S,5) -> (N,S,S,5)
        target_abs_box = IoU.relative2absolute(target[...,:5]) * target[...,4].unsqueeze(-1) #prevent noisy coordinates when confident=0
        prediction_abs_box1 = IoU.relative2absolute(prediction[...,:5])
        prediction_abs_box2 = IoU.relative2absolute(prediction[...,5:10])

        ### Find the best box: (N,S,S,4) -> (N*S*S,4) -> iou -> (N*S*S*4,1)
        iou_box1 = IoU.intersection_over_union(
            target_abs_box[...,:4].view(BATCH_SIZE*self.S*self.S,4), prediction_abs_box1[...,0:4].view(BATCH_SIZE*self.S*self.S,4)
            )
        iou_box2 = IoU.intersection_over_union(
            target_abs_box[...,:4].view(BATCH_SIZE*self.S*self.S,4), prediction_abs_box2[...,0:4].view(BATCH_SIZE*self.S*self.S,4)
            )

        ### Create a mask (N*S*S*4,1) -> (N,S,S,1) : if 0 -> box1, if 5 -> box2
        iou_mask = torch.gt(iou_box1, iou_box2).to(torch.int64)
        idx = iou_mask*5
        idx = idx.view(BATCH_SIZE, self.S, self.S, 1).repeat(1,1,1,5)
        idx = idx + torch.tensor([0,1,2,3,4])

        ### Retrieve predicted box coordinates and stack them regarding the best box : (N,S,S,1) -> (N,S,S,2)
        ### Clamp to prevent negative coordinates and sizes
        xywhc_hat = torch.gather(prediction, -1, idx)
        # xy_hat = xywhc_hat[...,:2].clamp(min=0)
        # wh_hat = xywhc_hat[...,2:4].clamp(min=0)
        xy_hat = xywhc_hat[...,:2]
        wh_hat = xywhc_hat[...,2:4]
        xy = target[...,:2]
        wh = target[...,2:4]
        
        ### Retrieve confidence numbers (N,S,S,1)
        # confidence_hat = xywhc_hat[...,4].unsqueeze(-1).clamp(min=0, max=1)
        confidence_hat = xywhc_hat[...,4].unsqueeze(-1)
        confidence = target[..., 4].unsqueeze(-1)

        ### Retrieve groundtruth class labels
        class_hat = prediction[..., 10:]
        class_true = target[...,5:]

        ### Create object mask : True if there is NO object (N,S,S,1)
        identity_obj = confidence.eq(0)

        ### Compute losses over the grid for the whole batch
        losses['loss_xy'] = self._MSELoss(xy_hat, xy, identity_obj)
        losses['loss_wh'] = self._MSELoss(wh_hat, wh, identity_obj, isRMSE=True)
        losses['loss_conf_obj'] = self._MSELoss(confidence_hat, confidence, identity_obj)
        losses['loss_conf_noobj'] = self._MSELoss(confidence_hat, confidence, torch.logical_not(identity_obj))
        losses['loss_class'] = self._MSELoss(class_hat, class_true, identity_obj)

        ### Compute the batch loss regarding the YoloV1 paper equation
        loss = self.LAMBD_COORD * losses['loss_xy'] \
                + self.LAMBD_COORD * losses['loss_wh'] \
                + losses['loss_conf_obj'] \
                + self.LAMBD_NOOBJ * losses['loss_conf_noobj'] \
                + losses['loss_class']

        assert torch.isnan(loss)==False, "Error : loss turns to NaN."
        return losses, loss