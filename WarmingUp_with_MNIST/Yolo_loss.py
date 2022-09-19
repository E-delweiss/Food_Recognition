import torch

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
        xc_hat, yc_hat = pred_coord_rcell.permute(1,0)
        xc, yc = true_coord_rcell.permute(1,0)

        xc_hat = torch.masked_fill(xc_hat, isObject, 0)
        yc_hat = torch.masked_fill(yc_hat, isObject, 0)
        xc = torch.masked_fill(xc, isObject, 0)
        yc = torch.masked_fill(yc, isObject, 0)

        squared_error = torch.pow(xc - xc_hat,2) + torch.pow(yc - yc_hat,2)
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
        rw_hat, rh_hat = pred_size.permute(1,0)
        rw, rh = true_size.permute(1,0)

        rw_hat = torch.masked_fill(rw_hat, isObject, 0)
        rh_hat = torch.masked_fill(rh_hat, isObject, 0)
        rw = torch.masked_fill(rw, isObject, 0)
        rh = torch.masked_fill(rh, isObject, 0)

        rw_hat = rw_hat.clamp(min=0)
        rh_hat = rh_hat.clamp(min=0)

        root_squared_error_w = torch.pow(torch.sqrt(rw) - torch.sqrt(rw_hat),2)
        root_squared_error_h = torch.pow(torch.sqrt(rh) - torch.sqrt(rh_hat),2)
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
            pred_class : torch.Tensor of shape (N, 10)
            true_class : torch.Tensor of shape (N, 10)
            isObject : torch.Bool of shape (N,1)
        Returns :
            squared_error : torch.Tensor of shape (N)
        """

        squared_error = torch.pow(true_class - pred_class, 2)
        squared_error = torch.sum(squared_error, dim=1)
        return torch.masked_fill(squared_error, isObject, 0) 

    def forward(self, pred_box:torch.Tensor, true_box:torch.Tensor, pred_class:torch.Tensor, true_class:torch.Tensor):
        """
        Grid forward pass.

        Args:
            pred_box : torch.Tensor of shape (N, S, S, 5)
                Batch predicted outputs containing xc_rcell, yc_rcell, rw, rh,
                and confident number c for each grid cell.
            true_box : torch.Tensor of shape (N, S, S, 5)
                Groundtrue batch containing bbox values for each cell and
                c indicate if there is an object to detect or not (1/0).
            pred_class : torch.Tensor of shape (N, S, S, 10)
                Probability of each digit class in each grid cell
            true_class : torch.Tensor of shape (N, 10)
                one-hot vect of each digit

        Return:
            loss : float
                The batch loss value of the grid
        """
        BATCH_SIZE = len(pred_box)

        ### Initialization of the losses
        losses_list = ['loss_xy', 'loss_wh', 'loss_conf_obj', 'loss_conf_noobj', 'loss_class']
        losses = {key : torch.zeros(BATCH_SIZE).to(self.device) for key in losses_list}
        
        ### Compute the losses for all images in the batch
        for i in range(self.S):
            for j in range(self.S):
                ### bbox coordinates
                xy_hat = pred_box[:,i,j,:2]
                xy = true_box[:,i,j,:2]
                wh_hat = pred_box[:,i,j,2:4]
                wh = true_box[:,i,j,2:4]
                
                ### confidence numbers
                pred_c = pred_box[:,i,j,4]
                true_c = true_box[:,i,j,4]

                ### objects to detect
                isObject = true_c.eq(0)

                ### sum the losses over the grid
                losses['loss_xy'] += self._coordloss(xy_hat, xy, isObject)
                losses['loss_wh'] += self._sizeloss(wh_hat, wh, isObject)
                losses['loss_conf_obj'] += self._confidenceloss(pred_c, true_c, isObject)
                losses['loss_conf_noobj'] += self._confidenceloss(pred_c, true_c, torch.logical_not(isObject))
                losses['loss_class'] += self._classloss(pred_class[:,i,j], true_class, isObject)

        
        
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


def test():
    criterion = YoloLoss(lambd_coord=5, lambd_noobj=0.5, S=6, device=torch.device('cpu'))
    prediction_box = torch.rand(32, 6, 6, 5)
    target_box = torch.rand(32, 6, 6, 5)
    prediction_label = torch.rand(32, 6, 6, 10)
    target_label = torch.rand(32, 10)

    losses, loss = criterion(prediction_box, target_box, prediction_label, target_label)
    print("Losses dict : ", losses.keys())
    print("Loss : ", loss)


if __name__ == "__main__":
    test()