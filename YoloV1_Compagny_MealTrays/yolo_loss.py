import torch
import IoU

class YoloLoss(torch.nn.Module):
    def __init__(self, lambd_coord:int, lambd_noobj:float, device:torch.device, S:int=6):
        super(YoloLoss, self).__init__()
        self.LAMBD_COORD = lambd_coord
        self.LAMBD_NOOBJ = lambd_noobj
        self.S = S
        self.device = device

    def _coordloss(self, pred_coord_rcell, true_coord_rcell, isObject):
        """
        Args : 
            pred_coord_rcell : torch.Tensor of shape (N, 2)
            true_coord_rcell : torch.Tensor of shape (N, 2)
            isObject : torch.Bool of shape (N,1)
        Returns :
            squared_error : torch.Tensor of shape (N)
        """
        xcr_hat, ycr_hat = pred_coord_rcell.permute(1,0)
        xcr, ycr = true_coord_rcell.permute(1,0)

        xcr_hat = torch.masked_fill(xcr_hat, isObject, 0)
        ycr_hat = torch.masked_fill(ycr_hat, isObject, 0)
        xcr = torch.masked_fill(xcr, isObject, 0)
        ycr = torch.masked_fill(ycr, isObject, 0)

        squared_error = torch.pow(xcr - xcr_hat,2) + torch.pow(ycr - ycr_hat,2)
        return squared_error

    def _sizeloss(self, pred_size, true_size, isObject):
        """
        Args :
            pred_size : torch.Tensor of shape (N, 2)
            true_size : torch.Tensor of shape (N, 2)
            isObject : torch.Bool of shape (N,1)
        Returns : 
            root_squared_error : torch.Tensor of shape (N)
        """
        wr_hat, hr_hat = pred_size.permute(1,0)
        wr, hr = true_size.permute(1,0)

        wr_hat = torch.masked_fill(wr_hat, isObject, 0)
        hr_hat = torch.masked_fill(hr_hat, isObject, 0)
        wr = torch.masked_fill(wr, isObject, 0)
        hr = torch.masked_fill(hr, isObject, 0)

        wr_hat = wr_hat.clamp(min=0)
        hr_hat = hr_hat.clamp(min=0)

        root_squared_error_w = torch.pow(torch.sqrt(wr) - torch.sqrt(wr_hat),2)
        root_squared_error_h = torch.pow(torch.sqrt(hr) - torch.sqrt(hr_hat),2)
        root_squared_error = root_squared_error_w + root_squared_error_h
        return root_squared_error

    def _confidenceloss(self, pred_c, true_c, isObject):
        """
        Args :
            pred_c : torch.Tensor of shape (N,1)
            true_c : torch.Tensor of shape (N,1)
            isObject : torch.Bool of shape (N,1)
        Return :
            squared_error : torch.Tensor of shape (N)
        """
        pred_c = torch.masked_fill(pred_c, isObject, 0)
        true_c = torch.masked_fill(true_c, isObject, 0)

        squared_error = torch.pow(true_c - pred_c, 2)
        return squared_error

    def _classloss(self, pred_class, true_class, isObject):
        """
        Args :
            pred_class : torch.Tensor of shape (N, C)
            true_class : torch.Tensor of shape (N, C)
            isObject : torch.Bool of shape (N,1)
        Returns :
            squared_error : torch.Tensor of shape (N)
        """
        squared_error = torch.pow(true_class - pred_class, 2)
        squared_error = torch.sum(squared_error, dim=1)
        return torch.masked_fill(squared_error, isObject, 0) 

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
        
        ### Compute the losses for all images in the batch
        for i in range(self.S):
            for j in range(self.S):                  
                xy_hat = prediction[:,i,j,:2]
                xy = target[:,i,j,:2]
                wh_hat = prediction[:,i,j,2:4]
                wh = target[:,i,j,2:4]
                
                ### confidence numbers
                pred_c = prediction[N, i, j, 4]
                true_c = target[N, i, j, 4]

                ### objects to detect
                isObject = true_c.eq(0)

                ### sum the losses over the grid
                losses['loss_xy'] += self._coordloss(xy_hat, xy, isObject)
                losses['loss_wh'] += self._sizeloss(wh_hat, wh, isObject)
                losses['loss_conf_obj'] += self._confidenceloss(pred_c, true_c, isObject)
                losses['loss_conf_noobj'] += self._confidenceloss(pred_c, true_c, torch.logical_not(isObject))

                ### class labels
                pred_class = prediction[N, i, j, 5:]
                true_class = target[N, i, j, 5:]
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
        return losses, loss