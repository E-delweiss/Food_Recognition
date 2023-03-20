import logging
from datetime import datetime

import torch
from tqdm import tqdm

def create_logging(prefix:str):
    """
    TODO

    Args:
        prefix (str): _description_
    """
    assert type(prefix) is str, TypeError

    log_format = (
    '%(asctime)s ::%(levelname)s:: %(message)s')

    tm = datetime.now()
    tm = tm.strftime("%d%m%Y_%Hh%M")
    logging_name = 'logging_'+prefix+'_'+tm+'.log'

    logging.basicConfig(
        level=logging.INFO,
        format=log_format, datefmt='%d/%m/%Y %H:%M:%S',
        filemode="w",
        filename=(logging_name),
    )
    logging.info("Model is {}.".format(prefix))

def device(verbose=0)->torch.device:
    """
    Set the device to 'cpu', 'cuda' or 'mps'.

    Args:
        None.

    Return:
        device : torch.device
    """

    ### Choosing device between CPU or GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.has_mps:
        device=torch.device('mps')
    else:
        device=torch.device('cpu')
    
    logging.info("Execute on {}".format(device))
    if verbose:
        print("\n------------------------------------")
        print(f"Execute script on - {device} -")
        print("------------------------------------\n")

    return device


def pretty_print(batch:int, BATCH_SIZE:int, len_training_ds:int, current_loss:float, losses:dict, train_classes_acc:float):
    """
    Print all training infos for the current batch.

    Args:
        batch (int)
            Current batch.
        BATCH_SIZE (int)
            Len of the batch size
        len_training_ds (int)
            Len of the training dataset.
        current_loss (float)
            Current training loss.
        losses (dict)
            Dict of all the losses used to compute the main loss. It contains floats :
            ['loss_xy', 'loss_wh', 'loss_conf_obj', 'loss_conf_noobj', 'loss_class'].
        train_classes_acc (float)
            Training class accuracy.
    """
    BATCH_SIZE = 128
    if batch+1 <= len_training_ds//BATCH_SIZE:
        current_training_sample = (batch+1)*BATCH_SIZE
    else:
        current_training_sample = batch*BATCH_SIZE + len_training_ds%BATCH_SIZE

    print(f"\n--- Image : {current_training_sample}/{len_training_ds}")
    print(f"* loss = {current_loss:.5f}")
    print(f"* xy_coord training loss for this batch : {losses['loss_xy']:.5f}")
    print(f"* wh_sizes training loss for this batch : {losses['loss_wh']:.5f}")
    print(f"* confidence with object training loss for this batch : {losses['loss_conf_obj']:.5f}")
    print(f"* confidence without object training loss for this batch : {losses['loss_conf_noobj']:.5f}")
    print(f"* class training loss for this batch : {losses['loss_class']:.5f}")
    print("\n")
    print(f"** Training class accuracy : {train_classes_acc*100:.2f}%")


def update_lr(current_epoch:int, optimizer:torch.optim):
    """
    Schedule the learning rate

    Args:
        current_epoch (int)
            Current training loop epoch.
        optimizer (torch.optim)
            Gradient descent optimizer.
    """
    if current_epoch > 7:
        optimizer.defaults['lr'] = 0.0001


def save_model(model, path:str, save:bool):
    """
    TODO

    Args:
        model (_type_): _description_
        path (str): _description_
        save (bool): _description_
    """
    if save:
        tm = datetime.now()
        tm = tm.strftime("%d%m%Y_%Hh%M")
        path = path+'_'+tm+'.pt'
        torch.save(model.state_dict(), path)
        print("\n")
        print("*"*5, "Model saved to {}.".format(path))
        logging.info("\nModel saved to {}.".format(path))
    else:
        return



def tqdm_fct(training_dataset):
    return tqdm(enumerate(training_dataset),
                total=len(training_dataset),
                initial=1,
                desc="Training : image",
                ncols=100)


def get_cells_with_object(tensor:torch.Tensor)->tuple:
    """
    TODO

    Args:
        tensor (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """
    assert type(tensor) is torch.Tensor, "Error: wrong type. Sould be torch.tensor"
    assert len(tensor.shape) == 4, "Error: tensor side should be (N,S,S,_). Ex: torch.Size([32, 7, 7, 5])"

    ### Get all indices with non zero values
    N, cells_i, cells_j, _ = tensor.to("cpu").nonzero().permute(1,0)

    ### Stacking in a new tensor
    cells_with_obj = torch.stack((N, cells_i, cells_j), dim=0)

    ### Get rid of same values
    N, cells_i, cells_j = torch.unique(cells_with_obj,dim=1)

    return N, cells_i, cells_j


def tensor2boxlist(tensor:torch.Tensor, B:int=1):
    """
    Turn tensor into list of boxes.
    tensor (N,S,S,6-11) -> list[img1[box1[x,y,w,h,c,label], ...], img2[...]]
    TODO
    """
    S = tensor.shape[1]

    tensor_old = tensor.clone()

    tensor = torch.zeros(1,S,S,5*B+1)
    tensor[...,:5*B] = tensor_old[...,:5*B]
    tensor[...,5*B] = torch.argmax(torch.softmax(tensor_old[...,5*B:], dim=-1), dim=-1)

    if B == 2 :
        tensor_box1 = tensor[...,:5].view(S*S, 5)
        tensor_box2 = tensor[...,5:10].view(S*S, 5)
        tensor_box = torch.concat((tensor_box1, tensor_box2), dim=0)
    else : 
        tensor_box = tensor[...,:5].view(S*S, 5)

    tensor_box = torch.concat((tensor_box, tensor[...,5*B].view(S*S, 1).repeat(B,1)),dim=-1)

    return tensor_box.tolist()





if __name__ == "__main__":
    create_logging()
    device = device()
    losses_list = ['loss_xy', 'loss_wh', 'loss_conf_obj', 'loss_conf_noobj', 'loss_class']
    losses = {key :0.5 for key in losses_list}
    pretty_print(7, 100, 0.87, losses, 65)
    a=7
    save_model(a, "toto_path", True)


