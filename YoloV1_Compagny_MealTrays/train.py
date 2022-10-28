import datetime
from timeit import default_timer as timer
import pickle
import logging

import torch

import utils
from yolo_loss import YoloLoss
from mealtrays_dataset import get_training_dataset, get_validation_dataset
# from darknet_like import YoloV1
# from darknet import YoloV1
# from tinydarknet import TinyYolo as YoloV1
# from darknet_like2 import YoloV1
from resnet101 import YoloV1
from metrics import MSE, MSE_confidenceScore, class_acc
from validation import validation_loop

learning_rate = 0.0001
BATCH_SIZE = 32
SAVE_MODEL = True
SAVE_LOSS = True
prefix="resnet101"
utils.create_logging(prefix=prefix)
device = utils.device(verbose=1)
logging.info(f"Learning rate = {learning_rate}")
logging.info(f"Batch size = {BATCH_SIZE}")

model = YoloV1(in_channels=3, S=7, C=8, B=2)
# model = YoloV1(448, S=7, C=8, B=2)
model = model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.0005)
loss_yolo = YoloLoss(lambd_coord=5, lambd_noobj=0.5, S=7, device=device)
logging.info(f"Using optimizer : {optimizer}")

training_dataloader = get_training_dataset(BATCH_SIZE, split="train", isNormalize=True, isAugment=True)
validation_dataloader = get_validation_dataset(split="test", isNormalize=True, isAugment=False)
DO_VALIDATION = True

################################################################################

delta_time = datetime.timedelta(hours=1)
timezone = datetime.timezone(offset=delta_time)

time_formatted = datetime.datetime.now(tz=timezone)
time_formatted = '{:%Y-%m-%d %H:%M:%S}'.format(time_formatted)
start_time = datetime.datetime.now()

print(f"[Training on] : {str(device).upper()}")
print(f"Learning rate : {optimizer.defaults['lr']}")
logging.info("Start training")
logging.info(f"[START] : {time_formatted}")

EPOCHS = 150
S = 7

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
    print(" "*5 + f"EPOCH {epoch+1}/{EPOCHS}")
    print(" "*5 + f"Learning rate : lr = {optimizer.defaults['lr']}")
    print("-"*20)

    #########################################################################

    for batch, (img, target) in utils.tqdm_fct(training_dataloader):
    # for batch in range(1):
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

        if batch == 0 or (batch+1)%8 == 0 or batch == len(training_dataloader.dataset)//BATCH_SIZE:
            # Recording the total loss
            batch_total_train_loss_list.append(current_loss)
            # Recording each losses 
            batch_train_losses_list.append(losses)
            # Recording class accuracy 
            batch_train_class_acc.append(train_classes_acc)

            utils.pretty_print(batch, len(training_dataloader.dataset), current_loss, losses, train_classes_acc, batch_size=BATCH_SIZE)

            ############### Compute validation metrics each 5 batch ###########################################
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
            #####################################################################################################

            if batch == len(training_dataloader.dataset)//BATCH_SIZE:
                print(f"Mean training loss for this epoch : {epochs_loss / len(training_dataloader):.5f}")
                print("\n\n")
                logging.info(f"Epoch {epoch+1}/{EPOCHS}")
                logging.info(f"***** Training loss : {epochs_loss / len(training_dataloader):.5f}")
                logging.info(f"***** MSE validation box loss : {mse_score:.5f}")
                logging.info(f"***** MSE validation confidence score : {mse_confidence_score:.5f}")
                logging.info(f"***** Validation class acc : {acc*100:.2f}%\n")
                
### Saving results
path_save_model = f"yoloPlato_{prefix}_{epoch+1}epochs"
utils.save_model(model, path_save_model, SAVE_MODEL)

pickle_val_results = {
"batch_val_MSE_box_list":batch_val_MSE_box_list,
"batch_val_confscore_list":batch_val_confscore_list,
"batch_val_class_acc":batch_val_class_acc
}

pickle_train_results = {
    "batch_train_losses_list" : batch_train_losses_list,
    "batch_train_class_acc" : batch_train_class_acc,
}

if SAVE_LOSS:
    with open('train_results.pkl', 'wb') as pkl:
        pickle.dump(pickle_train_results, pkl)

    with open('val_results.pkl', 'wb') as pkl:
        pickle.dump(pickle_val_results, pkl)

end_time = datetime.datetime.now()
logging.info('Time duration: {}.'.format(end_time - start_time))
logging.info("End training.")
#####################################################################################################
