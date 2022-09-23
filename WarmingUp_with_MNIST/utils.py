import torch

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
    
    print("\n------------------------------------")
    print(f"Execute notebook on - {device} -")
    print("------------------------------------\n")

    return device


def pretty_print(current_training_sample:int, len_training_ds:int, current_loss:float, losses:dict, train_classes_acc:float):
    """
    _summary_

    Args:
        current_training_sample (int): _description_
        len_training_ds (int): _description_
        current_loss (float): _description_
        losses (dict): _description_
        train_classes_acc (float): _description_
    """
    print(f"--- Image : {current_training_sample}/{len_training_ds}", f" : loss = {current_loss:.5f}")
    # print(f"xy_coord training loss for this batch : {torch.sum(losses['loss_xy']) / len(img):.5f}")
    print(f"xy_coord training loss for this batch : {losses['loss_xy']:.5f}")
    print(f"wh_sizes training loss for this batch : {losses['loss_wh']:.5f}")
    print(f"confidence with object training loss for this batch : {losses['loss_conf_obj']:.5f}")
    print(f"confidence without object training loss for this batch : {losses['loss_conf_noobj']:.5f}")
    print(f"class proba training loss for this batch : {losses['loss_class']:.5f}")
    print("\n")
    print(f"Training class accuracy : {train_classes_acc*100:.2f}%")


if __name__ == "__main__":
    device = device()
    losses_list = ['loss_xy', 'loss_wh', 'loss_conf_obj', 'loss_conf_noobj', 'loss_class']
    losses = {key :0.5 for key in losses_list}
    pretty_print(7, 100, 0.87, losses, 65)