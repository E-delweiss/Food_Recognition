import datetime
import logging
import os
import sys
from configparser import ConfigParser

import torch
from icecream import ic

import IoU
import mAP
import NMS
import utils
from yoloResnet import yoloResnet
from resnet50_old import resnet
from yoloModel import yoloModel
from mealtrays_dataset import get_training_dataset, get_validation_dataset
from metrics import class_acc
from validation import validation_loop
from loss import YoloLoss

################################################################################
current_folder = os.path.dirname(locals().get("__file__"))
config_file = os.path.join(current_folder, "config.ini")
sys.path.append(config_file)
################################################################################

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
S = config.getint('MODEL', 'grid_size')
B = config.getint('MODEL', 'nb_box')
C = config.getint('MODEL', 'nb_class')
PRETRAINED = config.getboolean('MODEL', 'pretrained_resnet')
LOAD_CHECKPOINT = config.getboolean('MODEL', 'yoloResnet_checkpoint')

isNormalize_trainset = config.getboolean('DATASET', 'isNormalize_trainset')
isAugment_trainset = config.getboolean('DATASET', 'isAugment_trainset')
isNormalize_valset = config.getboolean('DATASET', 'isNormalize_valset')
isAugment_valset = config.getboolean('DATASET', 'isAugment_valset')

LAMBD_COORD = config.getint('LOSS', 'lambd_coord')
LAMBD_NOOBJ = config.getfloat('LOSS', 'lambd_noobj')

PROB_THRESHOLD = config.getfloat('MAP', 'prob_threshold')
IOU_THRESHOLD = config.getfloat('MAP', 'iou_threshold')

FREQ = config.getint('PRINTING', 'freq')

################################################################################
device = utils.set_device(DEVICE, verbose=0)

# model = yoloResnet(pretrained=PRETRAINED, load_yoloweights=LOAD_CHECKPOINT, S=S, B=B, C=C)
model = yoloModel(pretrained=PRETRAINED, load_yoloweights=LOAD_CHECKPOINT, S=S, B=B, C=C)
# model = resnet(S=S, B=B, C=C)
model = model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
criterion = YoloLoss(lambd_coord=LAMBD_COORD, lambd_noobj=LAMBD_NOOBJ, S=S, device=device)

training_dataloader = get_training_dataset(BATCH_SIZE, split="train", isNormalize=isNormalize_trainset, isAugment=isAugment_trainset)
validation_dataloader = get_validation_dataset(split="test", isNormalize=isNormalize_valset, isAugment=isAugment_valset)

if LOAD_CHECKPOINT:
    pt_file = config.get('WEIGHTS', 'efficientnet_weights')
    ranger = utils.defineRanger(pt_file, EPOCHS)
else:
    ranger = range(EPOCHS)
################################################################################

delta_time = datetime.timedelta(hours=1)
timezone = datetime.timezone(offset=delta_time)

time_formatted = datetime.datetime.now(tz=timezone)
time_formatted = '{:%Y-%m-%d %H:%M:%S}'.format(time_formatted)
start_time = datetime.datetime.now()

print(f"[Training on] : {str(device).upper()}")
print(f"Learning rate : {optimizer.defaults['lr']}")

utils.create_logging(prefix=PREFIX)
logging.info(f"Pretrained is {PRETRAINED}")
if LOAD_CHECKPOINT: logging.info(f"RESTART FROM CHECKPOINT")
logging.info(f"Learning rate = {learning_rate}")
logging.info(f"Batch size = {BATCH_SIZE}")
logging.info(f"Using optimizer : {optimizer}")
logging.info("Lr Scheduler : None")
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

all_pred_boxes = []
all_true_boxes = []
for epoch in ranger:
    utils.update_lr(epoch, optimizer, LR_SCHEDULER)
    epochs_loss = 0.
    
    print("-"*20)
    print(" "*5 + f"EPOCH {epoch+1}/{EPOCHS+ranger[0]}")
    print(" "*5 + f"Learning rate : lr = {optimizer.defaults['lr']}")
    print("-"*20)

    ### Checkpoint
    if epoch % 50 == 0 and epoch != 0:
        utils.save_model(model, PREFIX+"CHECKPOINT_", epoch, SAVE_MODEL)

    ################################################################################

    for batch, (img, target) in utils.tqdm_fct(training_dataloader):
        model.train()
        loss = 0
        img, target = img.to(device), target.to(device)
        print("\n")

        ### clear gradients
        optimizer.zero_grad()
        
        ### prediction (N,S,S,B*(4+1)+C) -> (N,7,7,18)
        prediction = model(img)

        ### compute losses over each grid cell for each image in the batch
        losses, loss = criterion(prediction, target)
    
        ### compute gradients
        loss.backward()
        
        ### Weight updates
        optimizer.step()

        ##### Class accuracy
        train_class_acc, _ = class_acc(target, prediction)

        ######### print part #######################
        current_loss = loss.item()
        epochs_loss += current_loss

        if batch == 0 or (batch+1)%FREQ == 0 or batch == len(training_dataloader.dataset)//BATCH_SIZE:
            # Recording the total loss
            batch_total_train_loss_list.append(current_loss)
            # Recording each losses
            batch_train_losses_list.append(losses)
            # Recording class accuracy
            batch_train_class_acc.append(train_class_acc)

            utils.pretty_print(batch, len(training_dataloader.dataset), current_loss, losses, train_class_acc, batch_size=BATCH_SIZE)

            ############### Compute validation metrics each FREQ batch ###########################################
            if DO_VALIDATION:
                model.eval()
                train_idx = 0
                _, target_val, prediction_val = validation_loop(model, validation_dataloader, S, device)
                
                all_pred_boxes = []
                all_true_boxes = []
                for idx in range(len(target_val)):
                    true_bboxes = IoU.relative2absolute(target_val[idx].unsqueeze(0))
                    true_bboxes = utils.tensor2boxlist(true_bboxes)

                    nms_box_val = NMS.non_max_suppression(prediction_val[idx].unsqueeze(0), PROB_THRESHOLD, IOU_THRESHOLD)

                    for nms_box in nms_box_val:
                        all_pred_boxes.append([train_idx] + nms_box)

                    for box in true_bboxes:
                        # many will get converted to 0 pred
                        if box[4] > PROB_THRESHOLD:
                            all_true_boxes.append([train_idx] + box)
                    
                    train_idx += 1

                meanAP = mAP.mean_average_precision(all_true_boxes, all_pred_boxes, IOU_THRESHOLD)

                ### Validation accuracy
                acc, hard_acc = class_acc(target_val, prediction_val)

                batch_val_class_acc.append(acc)

                print(f"| Mean Average Precision @{IOU_THRESHOLD} : {meanAP:.2f}")
                print(f"| Validation class acc : {acc*100:.2f}%")
                print(f"| Validation class hard acc : {hard_acc*100:.2f}%")
                print("\n\n")
            else : 
                meanAP, acc, hard_acc = 9999, 9999, 9999
            ################################################################################

            if batch == len(training_dataloader.dataset)//BATCH_SIZE:
                print(f"Mean training loss for this epoch : {epochs_loss / len(training_dataloader):.5f}")
                print("\n\n")
                logging.info(f"Epoch {epoch+1}/{EPOCHS}")
                logging.info(f"***** Training loss : {epochs_loss / len(training_dataloader):.5f}")
                logging.info(f"***** Mean Average Precision @{IOU_THRESHOLD} : {meanAP:.2f}")
                logging.info(f"***** Validation class acc : {acc*100:.2f}%")
                logging.info(f"***** Validation class hard acc : {hard_acc*100:.2f}%\n")

                utils.save_model(model, PREFIX, epoch, SAVE_MODEL, time=False)
################################################################################
### Saving results
pickle_val_results = {
"batch_val_MSE_box_list":batch_val_MSE_box_list,
"batch_val_confscore_list":batch_val_confscore_list,
"batch_val_class_acc":batch_val_class_acc
}

pickle_train_results = {
    "batch_train_losses_list" : batch_train_losses_list,
    "batch_train_class_acc" : batch_train_class_acc,
}

utils.save_model(model, PREFIX, epoch, SAVE_MODEL)
utils.save_losses(pickle_train_results, pickle_val_results, PREFIX, SAVE_LOSS)

end_time = datetime.datetime.now()
logging.info('Time duration: {}.'.format(end_time - start_time))
logging.info("End training.")
################################################################################