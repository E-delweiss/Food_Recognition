import datetime
from timeit import default_timer as timer
import logging
from configparser import ConfigParser
import os, sys

import torch

import utils
from yolo_loss import YoloLoss
from mealtrays_dataset import get_training_dataset, get_validation_dataset
from darknet import darknet
from metrics import MSE, MSE_confidenceScore, class_acc
from validation import validation_loop

################################################################################

current_folder = os.path.dirname(locals().get("__file__"))
config_file = os.path.join(current_folder, "config.ini")
sys.path.append(config_file)

config = ConfigParser()
config.read('config.ini')

DEVICE = config.get('TRAINING', 'device')
learning_rate = config.getfloat('TRAINING', 'learning_rate')
BATCH_SIZE = config.getint('TRAINING', 'batch_size')
WEIGHT_DECAY = config.getfloat('TRAINING', 'weight_decay')
DO_VALIDATION = config.getboolean('TRAINING', 'do_validation')
EPOCHS = config.getint('TRAINING', 'nb_epochs')
LR_SCHEDULER = config.getboolean('TRAINING', 'lr_scheduler')

SAVE_MODEL = config.getboolean('SAVING', 'save_model')
SAVE_LOSS = config.getboolean('SAVING', 'save_loss')

PREFIX = config.get('MODEL', 'model_name')
IN_CHANNEL = config.getint('MODEL', 'in_channel')
S = config.getint('MODEL', 'grid_size')
B = config.getint('MODEL', 'nb_box')
C = config.getint('MODEL', 'nb_class')
PRETRAINED = config.getboolean('MODEL', 'pretrained')

isNormalize_trainset = config.getboolean('DATASET', 'isNormalize_trainset')
isAugment_trainset = config.getboolean('DATASET', 'isAugment_trainset')
isNormalize_valset = config.getboolean('DATASET', 'isNormalize_valset')
isAugment_valset = config.getboolean('DATASET', 'isAugment_valset')

LAMBD_COORD = config.getint('LOSS', 'lambd_coord')
LAMBD_NOOBJ = config.getfloat('LOSS', 'lambd_noobj')

FREQ = config.getint('PRINTING', 'freq')

################################################################################

device = utils.set_device(DEVICE, verbose=0)

model = darknet(pretrained=PRETRAINED, in_channel=IN_CHANNEL, S=S, B=B, C=C)
model = model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
loss_yolo = YoloLoss(lambd_coord=LAMBD_COORD, lambd_noobj=LAMBD_NOOBJ, S=S, device=device)

training_dataloader = get_training_dataset(BATCH_SIZE, split="train", isNormalize=isNormalize_trainset, isAugment=isAugment_trainset)
validation_dataloader = get_validation_dataset(split="test", isNormalize=isNormalize_valset, isAugment=isAugment_valset)

################################################################################

delta_time = datetime.timedelta(hours=1)
timezone = datetime.timezone(offset=delta_time)

time_formatted = datetime.datetime.now(tz=timezone)
time_formatted = '{:%Y-%m-%d %H:%M:%S}'.format(time_formatted)
start_time = datetime.datetime.now()

print(f"[Training on] : {str(device).upper()}")
print(f"Learning rate : {optimizer.defaults['lr']}")

utils.create_logging(prefix=PREFIX)
logging.info(f"Learning rate = {learning_rate}")
logging.info(f"Batch size = {BATCH_SIZE}")
logging.info(f"Using optimizer : {optimizer}")
logging.info("Lr Scheduler : lr/2 each 20 epochs")
logging.info("")
logging.info("Start training")
logging.info(f"[START] : {time_formatted}")

################################################################################
batch_total_train_loss_list = []
batch_train_losses_list = []
batch_train_class_acc = []

batch_val_MSE_box_list = []
batch_val_confscore_list = []
batch_val_class_acc = []

for epoch in range(EPOCHS):
    utils.update_lr(epoch, optimizer, LR_SCHEDULER)

    begin_time = timer()
    epochs_loss = 0.
    
    print("-"*20)
    print(" "*5 + f"EPOCH {epoch+1}/{EPOCHS}")
    print(" "*5 + f"Learning rate : lr = {optimizer.defaults['lr']}")
    print("-"*20)

    ################################################################################

    for batch, (img, target) in utils.tqdm_fct(training_dataloader):
        model.train()
        loss = 0
        begin_batch_time = timer()
        img, target = img.to(device), target.to(device)
        
        ### clear gradients
        optimizer.zero_grad()
        
        ### prediction (N,S,S,B*(4+1)+C) -> (N,7,7,18)
        prediction = model(img)
        
        ### compute losses over each grid cell for each image in the batch
        losses, loss = loss_yolo(prediction, target)
    
        ### compute gradients
        loss.backward()
        
        ### Weight updates
        optimizer.step()

        ##### Class accuracy
        train_classes_acc = class_acc(target, prediction)

        ######### print part #######################
        current_loss = loss.item()
        epochs_loss += current_loss

        if batch == 0 or (batch+1)%FREQ == 0 or batch == len(training_dataloader.dataset)//BATCH_SIZE:
            # Recording the total loss
            batch_total_train_loss_list.append(current_loss)
            # Recording each losses
            batch_train_losses_list.append(losses)
            # Recording class accuracy
            batch_train_class_acc.append(train_classes_acc)

            utils.pretty_print(batch, len(training_dataloader.dataset), current_loss, losses, train_classes_acc, batch_size=BATCH_SIZE)

            ############### Compute validation metrics each FREQ batch ###########################################
            if DO_VALIDATION:
                model.eval()
                _, target_val, prediction_val = validation_loop(model, validation_dataloader, S, device)
                
                ### Validation MSE score
                mse_score = MSE(target_val, prediction_val)

                ### Validation accuracy
                acc = class_acc(target_val, prediction_val)

                ### Validation confidence_score
                mse_confidence_score = MSE_confidenceScore(target_val, prediction_val)

                batch_val_MSE_box_list.append(mse_score)
                batch_val_confscore_list.append(mse_confidence_score)
                batch_val_class_acc.append(acc)

                print(f"| MSE validation box loss : {mse_score:.5f}")
                print(f"| MSE validation confidence score : {mse_confidence_score:.5f}")
                print(f"| Validation class acc : {acc*100:.2f}%")
                print("\n\n")
            else : 
                mse_score, mse_confidence_score, acc = 9999, 9999, 9999
            ################################################################################

            if batch == len(training_dataloader.dataset)//BATCH_SIZE:
                print(f"Mean training loss for this epoch : {epochs_loss / len(training_dataloader):.5f}")
                print("\n\n")
                logging.info(f"Epoch {epoch+1}/{EPOCHS}")
                logging.info(f"***** Training loss : {epochs_loss / len(training_dataloader):.5f}")
                logging.info(f"***** MSE validation box loss : {mse_score:.5f}")
                logging.info(f"***** MSE validation confidence score : {mse_confidence_score:.5f}")
                logging.info(f"***** Validation class acc : {acc*100:.2f}%\n")

################################################################################
### Saving results
path_save_model = f"yoloPlato_{PREFIX}_{epoch+1}epochs"

pickle_val_results = {
"batch_val_MSE_box_list":batch_val_MSE_box_list,
"batch_val_confscore_list":batch_val_confscore_list,
"batch_val_class_acc":batch_val_class_acc
}

pickle_train_results = {
    "batch_train_losses_list" : batch_train_losses_list,
    "batch_train_class_acc" : batch_train_class_acc,
}

utils.save_model(model, path_save_model, SAVE_MODEL)
utils.save_losses(pickle_train_results, pickle_val_results, PREFIX, SAVE_LOSS)

end_time = datetime.datetime.now()
logging.info('Time duration: {}.'.format(end_time - start_time))
logging.info("End training.")
################################################################################
