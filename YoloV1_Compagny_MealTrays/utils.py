import logging
import glob
from datetime import datetime
import pickle
from configparser import ConfigParser

from tqdm import tqdm
import PIL
import torch
import torchvision

def create_logging(prefix:str):
    """
    Create logging file.

    Args:
        prefix (str)
    """
    assert type(prefix) is str, TypeError

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

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
    logging.info(f"Model is {prefix}")
    


def set_device(device, verbose=0)->torch.device:
    """
    Set the device to 'cpu', 'cuda' or 'mps'.

    Args:
        None.
    Return:
        device : torch.device
    """
    if device == 'cpu':
        device = torch.device('cpu')

    if device == 'cuda' and torch.cuda.is_available():
        torch.device('cuda')
    elif device == 'cuda' and torch.cuda.is_available() is False:
        logging.warning(f"Device {device} not available.")

    if device == 'mps' and torch.has_mps:
        logging.warning(f"Device {device} currently not working well with PyTorch.")
        # device = torch.device('cpu')

    logging.info("Execute on {}".format(device))
    if verbose:
        print("\n------------------------------------")
        print(f"Execute script on - {device} -")
        print("------------------------------------\n")

    return device


def pretty_print(batch:int, len_training_ds:int, current_loss:float, losses:dict, train_classes_acc:float, batch_size:int):
    """
    Print all training infos for the current batch.

    Args:
        batch (int)
            Current batch.
        len_training_ds (int)
            Len of the training dataset.
        current_loss (float)
            Current training loss.
        losses (dict)
            Dict of all the losses used to compute the main loss. It contains floats :
            ['loss_xy', 'loss_wh', 'loss_conf_obj', 'loss_conf_noobj', 'loss_class'].
        train_classes_acc (float)
            Training class accuracy.
        batch_size (int)
            Nb of image in each batch (the last one may be smaller)
    """
    BATCH_SIZE = batch_size
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


def update_lr(current_epoch:int, optimizer:torch.optim, do_scheduler:bool):
    """
    Schedule the learning rate

    Args:
        current_epoch (int)
            Current training loop epoch.
        optimizer (torch.optim)
            Gradient descent optimizer.
        do_scheduler (bool)
            TODO
    """
    # k = 0.1
    # optimizer.defaults['lr'] = k/np.sqrt(current_epoch+1) * lr0
    # logging.info(f"Learning rate : lr {optimizer.defaults['lr']}")

    if do_scheduler:
        if current_epoch < 30:
            lr = [0.0001 + x * (0.001 - 0.0001)/30 for x in range(30)]
            optimizer.defaults['lr'] = lr[current_epoch]
        if current_epoch >= 80:
            optimizer.defaults['lr'] = optimizer.defaults['lr']/2
        # if current_epoch % 20 == 0 and current_epoch != 0:
        #     optimizer.defaults['lr'] = optimizer.defaults['lr']/2
        logging.info(f"Learning rate : lr {optimizer.defaults['lr']}")


def defineRanger(pt_file:str, num_epoch:int)->range:
    """
    Create a ranger for the training loop. 
    Handle a start != 0 for checkpoint.

    Args:
        pt_file (str)
            Pytorch checkpoint file.
        num_epoch (int)
            Number of epochs (modify the epoch to start with)
    """
    # start_epoch = int(pt_file[:pt_file.find("epochs")][-2:])
    start_epoch = ''.join(c for c in pt_file.split("_")[1] if c.isdigit())
    start_epoch = int(start_epoch)
    end_epoch = int(start_epoch) + num_epoch
    ranger = range(start_epoch, end_epoch+1)
    return ranger


def save_model(model, prefix:str, current_epoch:int, save:bool, time=True):
    """
    Save Pytorch weights of the model. Set the name based on timeclock.

    Args:
        model (torch model)
            Training model.
        prefix (str)
            Used to create the pt file name.
        current_epoch (int)
            Used to create the pt file name.
        save (bool)
            If False, set a warning in log file.
    """
    if save:
        path = f"{prefix}_{current_epoch+1}epochs"
        tm = datetime.now()
        tm = tm.strftime("%d%m%Y_%Hh%M")
        if time:
            path = path+'_'+tm+'.pt'
        else:
            path = path+'.pt' 
        torch.save(model.state_dict(), path)
        print("\n")
        print("*"*5, "Model saved to {}.".format(path))
        logging.info("\n")
        logging.info("Model saved to {}.".format(path))
    else:
        logging.warning("No saving has been requested for model.")
    return

def save_losses(train_loss:dict, val_loss:dict, model_name:str, save:bool):
    """
    Save training en validation losses to pickle files.

    Args:
        train_loss (dict)
        val_loss (dict)
        model_name (str)
        save (bool)
    """
    if save:
        tm = datetime.now()
        tm = tm.strftime("%d%m%Y_%Hh%M")
        train_path = f"train_results_{model_name}_{tm}.pkl"
        val_path = f"val_results_{model_name}_{tm}.pkl"
        
        with open(train_path, 'wb') as pkl:
            pickle.dump(train_loss, pkl)

        with open(val_path, 'wb') as pkl:
            pickle.dump(val_loss, pkl)
        
        logging.info("Training results saved to {}.".format(train_path))
        logging.info("Validation results saved to {}.".format(val_path))
    else:
        logging.warning("No saving has been requested for losses.")
    return


def tqdm_fct(training_dataset):
    """
    Set a tqdm progress bar.
    """
    return tqdm(enumerate(training_dataset),
                total=len(training_dataset),
                initial=1,
                desc="Training : image",
                ncols=100)


def mean_std_normalization()->tuple:
    """
    Get the mean and std of the dataset RGB channels.
    Note:
        mean/std of the whole dataset :
            mean=(0.4111, 0.4001, 0.3787), std=(0.3461, 0.3435, 0.3383)
        mean/std of the dataset labellised only (12/10/22) :
            mean=(0.4168, 0.4055, 0.3838), std=(0.3475, 0.3442, 0.3386)

    Returns:
        mean : torch.Tensor
        std : torch.Tensor
    """
    data_jpg = glob.glob('YoloV1_Compagny_MealTrays/mealtrays_dataset/obj_train_data/*.jpg')
    data_PIL = [PIL.Image.open(img_path).convert('RGB') for img_path in data_jpg]
    data_tensor = [torchvision.transforms.ToTensor()(img_PIL) for img_PIL in data_PIL]

    channels_sum, channels_squared_sum = 0, 0
    for img in data_tensor:
        channels_sum += torch.mean(img, dim=[1,2])
        channels_squared_sum += torch.mean(img**2, dim=[1,2])
    
    mean = channels_sum/len(data_tensor)
    std = torch.sqrt((channels_squared_sum/len(data_tensor) - mean**2))
    
    return mean, std


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


def tensor2boxlist(tensor:torch.Tensor):
    """
    Turn tensor into list of boxes.
    tensor (N,S,S,6-11) -> list[img1[box1[x,y,w,h,c,label], ...], img2[...]]
    TODO
    """
    C = 8
    S = 7
    B = (tensor.shape[-1] - C) // 5
    tensor_old = tensor.clone()

    tensor = torch.zeros(1,7,7,5*B+1)
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
    mean, std = mean_std_normalization()
    print(mean)
    print(std)