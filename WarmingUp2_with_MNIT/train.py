import datetime
from timeit import default_timer as timer
import pickle
import logging

import torch

import utils
from yolo_loss import YoloLoss
from MNIST_dataset import get_training_dataset, get_validation_dataset
from smallnet import netMNIST
from metrics import class_acc
from validation import validation_loop
import NMS, mAP

learning_rate = 0.001
BATCH_SIZE = 32
SAVE_MODEL = True
prefix="smallnet"
utils.create_logging(prefix=prefix)
device = utils.device(verbose=1)
logging.info(f"Learning rate = {learning_rate}")
logging.info(f"Batch size = {BATCH_SIZE}")
                
model_MNIST = netMNIST(sizeHW=140, S=6, B=2, C=10)
model_MNIST = model_MNIST.to(device)
optimizer = torch.optim.Adam(params=model_MNIST.parameters(), lr=learning_rate, weight_decay=0.0005)
loss_yolo = YoloLoss(lambd_coord=5, lambd_noobj=0.5, S=6, device=device)
logging.info(f"Using optimizer : {optimizer}")

training_dataset = get_training_dataset()
validation_dataset = get_validation_dataset(32)

################################################################################

delta_time = datetime.timedelta(hours=1)
timezone = datetime.timezone(offset=delta_time)

t = datetime.datetime.now(tz=timezone)
str_t = '{:%Y-%m-%d %H:%M:%S}'.format(t)
start_time = datetime.datetime.now()

print(f"[START] : {str_t} :")
print(f"[Training on] : {str(device).upper()}")
print(f"Learning rate : {optimizer.defaults['lr']}")
logging.info("Start training")
logging.info(f"[START] : {str_t}")

EPOCHS = 10
S = 6
C = 10
B=2
frame_size = 140
PROB_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

################################################################################
batch_total_train_loss_list = []
batch_train_losses_list = []
batch_train_class_acc = []

batch_val_MSE_box_list = []
batch_val_confscore_list = []
batch_val_class_acc = []

for epoch in range(EPOCHS):
    utils.update_lr(epoch, optimizer)

    begin_time = timer()
    epochs_loss = 0.
    
    print("-"*20)
    str_t = '{:%Y-%m-%d %H:%M:%S}'.format(t)
    print(" "*5 + f"{str_t} : EPOCH {epoch+1}/{EPOCHS}")
    print(" "*5 + f"Learning rate : lr = {optimizer.defaults['lr']}")
    print("-"*20)

    #########################################################################

    for batch, (img, target) in utils.tqdm_fct(training_dataset):
        model_MNIST.train()
        loss = 0
        begin_batch_time = timer()
        img, target = img.to(device), target.to(device)
        
        ### clear gradients
        optimizer.zero_grad()
        
        ### prediction (N,S,S,5) & (N,S,S,10)
        prediction = model_MNIST(img)
        
        ### compute losses over each grid cell for each image in the batch
        losses, loss = loss_yolo(prediction, target)
    
        ### compute gradients
        loss.backward()
        
        ### Weight updates
        optimizer.step()

        ##### Class accuracy
        # train_classes_acc = class_acc(bbox_true, labels, label_preds)
        train_classes_acc = 999999

        ######### print part #######################
        current_loss = loss.item()
        epochs_loss += current_loss

        if batch == 0 or (batch+1)%100 == 0 or batch == len(training_dataset.dataset)//BATCH_SIZE:
            # Recording the total loss
            batch_total_train_loss_list.append(current_loss)
            # Recording each losses 
            batch_train_losses_list.append(losses)
            # Recording class accuracy 
            batch_train_class_acc.append(train_classes_acc)

            utils.pretty_print(batch, len(training_dataset.dataset), current_loss, losses, train_classes_acc)

            ############### Compute validation metrics each 100 batch ###########################################
            if DO_VALIDATION:
                train_idx = 0
                _, target_val, prediction_val = validation_loop(model_MNIST, validation_dataloader, S, device)
                
                all_pred_boxes = []
                all_true_boxes = []
                for idx in range(len(target_val)):
                    true_bboxes = IoU.relative2absolute(target_val[idx].unsqueeze(0), frame_size)
                    true_bboxes = utils.tensor2boxlist(true_bboxes)

                    nms_box_val = NMS.non_max_suppression(prediction_val[idx].unsqueeze(0), frame_size, PROB_THRESHOLD, IOU_THRESHOLD)

                    for nms_box in nms_box_val:
                        all_pred_boxes.append([train_idx] + nms_box)

                    for box in true_bboxes:
                        # many will get converted to 0 pred
                        if box[4] > PROB_THRESHOLD:
                            all_true_boxes.append([train_idx] + box)
                    
                    train_idx += 1

                meanAP = mAP.mean_average_precision(all_true_boxes, all_pred_boxes, S, C, IOU_THRESHOLD)

                ### Validation accuracy
                acc, hard_acc = class_acc(target_val, prediction_val)

                batch_val_class_acc.append(acc)

                print(f"| Mean Average Precision @{IOU_THRESHOLD} : {meanAP:.2f}")
                print(f"| Validation class acc : {acc*100:.2f}%")
                print(f"| Validation class hard acc : {hard_acc*100:.2f}%")
                print("\n\n")
            else : 
                meanAP, acc, hard_acc = 9999, 9999, 9999

            ### Validation MSE score
            # mse_score = MSE(bbox_true, bbox_preds)
            mse_score = 999999

            ### Validation accuracy
            # acc = class_acc(bbox_true, labels, label_preds)
            acc = 99999

            ### Validation confidence_score
            # mse_confidence_score = MSE_confidenceScore(bbox_true, bbox_preds)
            mse_confidence_score = 999999

            batch_val_MSE_box_list.append(mse_score)
            batch_val_confscore_list.append(mse_confidence_score)
            batch_val_class_acc.append(acc)

            print(f"| MSE validation box loss : {mse_score:.5f}")
            print(f"| MSE validation confidence score : {mse_confidence_score:.5f}")
            print(f"| Validation class acc : {acc*100:.2f}%")
            print("\n\n")
            #####################################################################################################

            if batch == len(training_dataset.dataset)//BATCH_SIZE:
                print(f"Total elapsed time for training : {datetime.timedelta(seconds=timer()-begin_time)}")
                print(f"Mean training loss for this epoch : {epochs_loss / len(training_dataset):.5f}")
                print("\n\n")
                logging.info(f"Epoch {epoch+1}/{EPOCHS}")
                logging.info(f"***** Training loss : {epochs_loss / len(training_dataset):.5f}")
                logging.info(f"***** MSE validation box loss : {mse_score:.5f}")
                logging.info(f"***** MSE validation confidence score : {mse_confidence_score:.5f}")
                logging.info(f"***** Validation class acc : {acc*100:.2f}%")

### Saving results
path_save_model = f"MNIST_{prefix}_{epoch+1}epochs"
utils.save_model(model_MNIST, path_save_model, SAVE_MODEL)

pickle_val_results = {
"batch_val_MSE_box_list":batch_val_MSE_box_list,
"batch_val_confscore_list":batch_val_confscore_list,
"batch_val_class_acc":batch_val_class_acc
}

pickle_train_results = {
    "batch_train_losses_list" : batch_train_losses_list,
    "batch_train_class_acc" : batch_train_class_acc,
}

with open('train_results.pkl', 'wb') as pkl:
    pickle.dump(pickle_train_results, pkl)

with open('val_results.pkl', 'wb') as pkl:
    pickle.dump(pickle_val_results, pkl)

end_time = datetime.datetime.now()
logging.info('Time duration: {}.'.format(end_time - start_time))
logging.info("End training.")
#####################################################################################################
