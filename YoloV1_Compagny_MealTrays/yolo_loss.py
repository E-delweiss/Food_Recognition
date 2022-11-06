import torch
import IoU

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
        mse = torch.nn.MSELoss(reduce='mean')
        rmse = torch.sqrt(mse(true_size,pred_size) + eps)
        return rmse

    def _confidenceloss(self, pred_c, true_c, isObject):
        """
        TODO
        """
        isObject = isObject.repeat(1,1,1,pred_c.shape[-1])
        pred_c = pred_c.clamp(min=0, max=1)
        pred_c = torch.masked_fill(pred_c, isObject, 0)
        
        mse = torch.nn.MSELoss(reduce='mean')
        return mse(true_c, pred_c)

    def _classloss(self, pred_class, true_class, isObject):
        """
        TODO
        """
        isObject = isObject.repeat(1,1,1,pred_class.shape[-1])
        torch.masked_fill(pred_class, isObject, 0) 
        mse = torch.nn.MSELoss(reduce='mean')
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
        N = range(BATCH_SIZE)

        ### Initialization of the losses
        losses_list = ['loss_xy', 'loss_wh', 'loss_conf_obj', 'loss_conf_noobj', 'loss_class']
        losses = {key : torch.zeros(BATCH_SIZE).to(self.device) for key in losses_list}
        
        target_abs_box = IoU.relative2absolute(target[:,:,:,:5]) # -> (N,4)
        prediction_abs_box1 = IoU.relative2absolute(prediction[...,:5])
        prediction_abs_box2 = IoU.relative2absolute(prediction[...,5:10])

        iou_box1 = IoU.intersection_over_union(
            target_abs_box[...,:4].view(BATCH_SIZE*self.S*self.S,4), prediction_abs_box1[:,:,:,0:4].view(BATCH_SIZE*self.S*self.S,4)
            ) # -> (N,1)
        iou_box2 = IoU.intersection_over_union(
            target_abs_box[...,:4].view(BATCH_SIZE*self.S*self.S,4), prediction_abs_box2[...,5:8].view(BATCH_SIZE*self.S*self.S,4)
            ) # -> (N,1)
        iou_mask = torch.gt(iou_box1, iou_box2).to(torch.int64)

        idx = 5*iou_mask #if 0 -> box1 infos, if 5 -> box2 infos

        print("\nDEBUG : ", idx.view(BATCH_SIZE,self.S,self.S)[4,2,3], idx.view(BATCH_SIZE,self.S,self.S).shape)
        print("\n")

        ### bbox coordinates
        x_hat = prediction[:, :, :, idx]
        y_hat = prediction[:, :, :, idx+1]
        w_hat = prediction[:, :, :, idx+2]
        h_hat = prediction[:, :, :, idx+3]

        xy_hat = torch.stack((x_hat, y_hat), dim=-1)
        wh_hat = torch.stack((w_hat, h_hat), dim=-1)

        xy = target[...,:2]
        wh = target[...,2:4]
        
        ### confidence numbers
        confidence_hat = prediction[:, :, :, idx+4]
        confidence = target[:, :, :, 4]

        ### objects to detect
        identity_obj = target.eq(0).unsqueeze(-1)

        ### sum the losses over the grid
        losses['loss_xy'] = self._coordloss(xy_hat, xy, identity_obj)
        losses['loss_wh'] = self._sizeloss(wh_hat, wh, identity_obj)
        losses['loss_conf_obj'] = self._confidenceloss(confidence_hat, confidence, identity_obj)
        losses['loss_conf_noobj'] = self._confidenceloss(confidence_hat, confidence, torch.logical_not(identity_obj))

        ### class labels
        pred_class = prediction[..., 10:]
        true_class = target[...,5:]
        losses['loss_class'] = self._classloss(pred_class, true_class, identity_obj)

        loss = self.LAMBD_COORD * losses['loss_xy'] \
                + self.LAMBD_COORD * losses['loss_wh'] \
                + losses['loss_conf_obj'] \
                + self.LAMBD_NOOBJ * losses['loss_conf_noobj'] \
                + losses['loss_class']

        # assert torch.isnan(loss)==False, "Error : loss turns to NaN."   
        return losses, loss


if False:
#################################
        ### Compute the losses for all images in the batch
        for cell_i in range(self.S):
            for cell_j in range(self.S):
                iou_box = []
                target_box_abs = IoU.relative2absolute(target[:,:,:,:5], N, cell_i, cell_j) # -> (N,4)
                for b in range(self.B):
                    box_k = 5*b
                    prediction_box_abs = IoU.relative2absolute(prediction[:,:,:, box_k : 5+box_k], N, cell_i, cell_j) # -> (N,4)
                    iou = IoU.intersection_over_union(target_box_abs[...,:4], prediction_box_abs[...,:4]) # -> (N,1)
                    iou_box.append(iou) # -> [iou_box1:(N), iou_box:(N)]
                
                ### TODO comment
                iou_mask = torch.gt(iou_box[0], iou_box[1])
                iou_mask = iou_mask.to(torch.int64)
                idx = 5*iou_mask #if 0 -> box1 infos, if 5 -> box2 infos

                ### bbox coordinates relating to the box with the largest IoU
                ### note : python doesn't like smth like a[N,i,j, arr1:arr2]
                x_hat = prediction[N, cell_i, cell_j, idx]
                y_hat = prediction[N, cell_i, cell_j, idx+1]
                w_hat = prediction[N, cell_i, cell_j, idx+2]
                h_hat = prediction[N, cell_i, cell_j, idx+3]
                
                xy_hat = torch.stack((x_hat, y_hat), dim=-1)
                wh_hat = torch.stack((w_hat, h_hat), dim=-1)
                
                xy = target[N, cell_i, cell_j, :2]
                wh = target[N, cell_i, cell_j, 2:4]
                
                ### confidence numbers
                pred_c = prediction[N, cell_i, cell_j, idx+4]
                true_c = target[N, cell_i, cell_j, 4]

                ### objects to detect
                isObject = true_c.eq(0)

                ### sum the losses over the grid
                losses['loss_xy'] += self._coordloss(xy_hat, xy, isObject)
                losses['loss_wh'] += self._sizeloss(wh_hat, wh, isObject)
                losses['loss_conf_obj'] += self._confidenceloss(c_hat, true_c, isObject)
                losses['loss_conf_noobj'] += self._confidenceloss(c_hat, true_c, torch.logical_not(isObject))

                ### class labels
                pred_class = prediction[N, cell_i, cell_j, 5*self.B:]
                true_class = target[N, cell_i, cell_j, 5:]
                losses['loss_class'] += self._classloss(pred_class, true_class, isObject)
     
        ### Yolo_v1 loss over the batch, shape : (BATCH_SIZE)
        for key, value in losses.items():
            losses[key] = torch.sum(value)/BATCH_SIZE

        loss = self.LAMBD_COORD * losses['loss_xy'] \
                + self.LAMBD_COORD * losses['loss_wh'] \
                + losses['loss_conf_obj'] \
                + self.LAMBD_NOOBJ * losses['loss_conf_noobj'] \
                + losses['loss_class']

        loss = torch.sum(loss) / BATCH_SIZE

        assert torch.isnan(loss)==False, "Error : loss turns to NaN."   
