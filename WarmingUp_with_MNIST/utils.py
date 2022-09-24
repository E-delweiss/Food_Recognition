import logging
import torch
from tqdm import tqdm

def create_logging():
    log_format = (
    '%(asctime)s ::%(levelname)s:: %(message)s')

    logging.basicConfig(
        level=logging.INFO,
        format=log_format, datefmt='%d/%m/%Y %H:%M:%S',
        filemode="w",
        filename=('logging.log'),
    )

def device()->torch.device:
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
    # elif torch.has_mps:
    #     device=torch.device('mps')
    else:
        device=torch.device('cpu')
    
    logging.info("Execute on {}".format(device))
    print("\n------------------------------------")
    print(f"Execute script on - {device} -")
    print("------------------------------------\n")

    return device


def pretty_print(batch:int, len_training_ds:int, current_loss:float, losses:dict, train_classes_acc:float):
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
    """
    BATCH_SIZE = 64
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

def save_model(model, path, save):
    from datetime import datetime
    if not save:
        return
    tm = datetime.now()
    tm = tm.strftime("%d%m%Y_%Hh%M")
    path = path+'_'+tm+'.pt'
    # torch.save(model.state_dict(), path)
    print("\n")
    print("*"*5, "Model saved to {}.".format(path))



def tqdm_fct(training_dataset):
    return tqdm(enumerate(training_dataset),
                total=len(training_dataset),
                initial=1,
                desc="Training : image",
                ncols=100)


if __name__ == "__main__":
    create_logging()
    device = device()
    losses_list = ['loss_xy', 'loss_wh', 'loss_conf_obj', 'loss_conf_noobj', 'loss_class']
    losses = {key :0.5 for key in losses_list}
    pretty_print(7, 100, 0.87, losses, 65)
    a=7
    save_model(a, "toto_path", True)


