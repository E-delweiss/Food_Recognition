import logging
import glob

from tqdm import tqdm
import PIL
import torch
import torchvision

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
    torch.save(model.state_dict(), path)
    print("\n")
    print("*"*5, "Model saved to {}.".format(path))



def tqdm_fct(training_dataset):
    return tqdm(enumerate(training_dataset),
                total=len(training_dataset),
                initial=1,
                desc="Training : image",
                ncols=100)


def mean_std_normalization()->tuple:
    """
    TODO

    Returns:
        mean : torch.Tensor
            TODO
        std : torch.Tensor
            TODO
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

if __name__ == "__main__":
    mean, std = mean_std_normalization()
    print(mean)
    print(std)


