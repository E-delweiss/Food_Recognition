import torch
import IoU
from icecream import ic

class YoloLoss(torch.nn.Module):
    def __init__(self, lambd_coord:int, lambd_noobj:float, device:torch.device, S:int=7, B:int=2):
        super(YoloLoss, self).__init__()
        self.LAMBD_COORD = lambd_coord
        self.LAMBD_NOOBJ = lambd_noobj
        self.S = S
        self.B = B
        self.device = device

    def _coordloss(self, pred_coord, true_coord, isObject):
        """
        TODO
        """
        isObject = isObject.repeat(1,1,1,pred_coord.shape[-1])

        pred_coord = pred_coord.clamp(min=0)
        pred_coord = torch.masked_fill(pred_coord, isObject, 0)

        mse = torch.nn.MSELoss(reduction='mean')
        return mse(true_coord, pred_coord)

    def _sizeloss(self, pred_size, true_size, isObject):
        """
        TODO
        """
        isObject = isObject.repeat(1,1,1,pred_size.shape[-1])

        pred_size = pred_size.clamp(min=0)
        pred_size = torch.masked_fill(pred_size, isObject, 0)

        eps = 1e-6
        mse = torch.nn.MSELoss(reduction ='mean')
        rmse = torch.sqrt(mse(true_size,pred_size) + eps)
        return rmse

    def _confidenceloss(self, pred_c, true_c, isObject):
        """
        TODO
        """
        isObject = isObject.repeat(1,1,1,pred_c.shape[-1])

        pred_c = pred_c.clamp(min=0, max=1)
        pred_c = torch.masked_fill(pred_c, isObject, 0)
        
        mse = torch.nn.MSELoss(reduction ='mean')
        return mse(true_c, pred_c)

    def _classloss(self, pred_class, true_class, isObject):
        """
        TODO
        """
        isObject = isObject.repeat(1,1,1,pred_class.shape[-1])
        torch.masked_fill(pred_class, isObject, 0)
        mse = torch.nn.MSELoss(reduction ='mean')
        return mse(true_class, pred_class)

    def forward(self, prediction:torch.Tensor, target:torch.Tensor):
        """
        Grid fowrard pass.

        Args:
            prediction : torch.Tensor of shape (N, S, S, B*5 + C)
                Batch predicted outputs containing 2 box infos (xcr_rcell, ycr_rcell, wr, hr, c)
                and a one-hot encoded class for each grid cell.

            target : torch.Tensor of shape (N, S, S, 5 + C)
                Batch groundtrouth outputs containing xcr_rcell, ycr_rcell, wr, hr,
                confident number c and one-hot encoded class for each grid cell.

        Return:
            loss : float
                The batch loss value of the grid
        """
        BATCH_SIZE = len(prediction)

        ### Initialization of the losses
        losses_list = ['loss_xy', 'loss_wh', 'loss_conf_obj', 'loss_conf_noobj', 'loss_class']
        losses = {key : torch.zeros(BATCH_SIZE).to(self.device) for key in losses_list}
        
        ### Get absolute coordinates: (N,S,S,5) -> (N,S,S,5)
        target_abs_box = IoU.relative2absolute(target[...,:5])
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
        idx = idx.view(BATCH_SIZE, self.S, self.S, 1)

        ### Retrieve predicted box coordinates and stack them regarding the best box : (N,S,S,1) -> (N,S,S,2)
        x_hat = torch.gather(prediction, -1, idx)
        y_hat = torch.gather(prediction, -1, idx+1)
        w_hat = torch.gather(prediction, -1, idx+2)
        h_hat = torch.gather(prediction, -1, idx+3)

        xy_hat = torch.concat((x_hat, y_hat), dim=-1)
        wh_hat = torch.concat((w_hat, h_hat), dim=-1)

        ### Retrieve groundtruths box coordinates (N,S,S,2)
        xy = target[...,:2]
        wh = target[...,2:4]
        
        ### Retrieve confidence numbers (N,S,S,1)
        confidence_hat = torch.gather(prediction, -1, idx+4)
        confidence = target[..., 4].unsqueeze(-1)

        ### Create object mask : True if there is NO object (N,S,S,1,1)
        identity_obj = confidence.eq(0)

        ### Compute losses over the grid for the all batch
        losses['loss_xy'] = self._coordloss(xy_hat, xy, identity_obj)
        losses['loss_wh'] = self._sizeloss(wh_hat, wh, identity_obj)
        losses['loss_conf_obj'] = self._confidenceloss(confidence_hat, confidence, identity_obj)
        losses['loss_conf_noobj'] = self._confidenceloss(confidence_hat, confidence, torch.logical_not(identity_obj))

        ### class labels
        class_hat = prediction[..., 10:]
        class_true = target[...,5:]
        losses['loss_class'] = self._classloss(class_hat, class_true, identity_obj)

        loss = self.LAMBD_COORD * losses['loss_xy'] \
                + self.LAMBD_COORD * losses['loss_wh'] \
                + losses['loss_conf_obj'] \
                + self.LAMBD_NOOBJ * losses['loss_conf_noobj'] \
                + losses['loss_class']

        # assert torch.isnan(loss)==False, "Error : loss turns to NaN."   
        return losses, loss