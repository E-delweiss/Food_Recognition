import datetime
from timeit import default_timer as timer
import torch

from utils import device, pretty_print, update_lr, save_model, tqdm_fct
from Yolo_loss import YoloLoss
from MNIST_dataset import get_training_dataset, get_validation_dataset
from Darknet_like import YoloMNIST
from Metrics import class_acc
from Validation import validation_loop

learning_rate = 0.001
BATCH_SIZE = 64
SAVE_MODEL = False
device = device() 

model_MNIST = YoloMNIST(sizeHW=75, S=6, C=10, B=1)
model_MNIST = model_MNIST.to(device)
optimizer = torch.optim.Adam(params=model_MNIST.parameters(), lr=learning_rate, weight_decay=0.0005)
loss_yolo = YoloLoss(lambd_coord=5, lambd_noobj=0.5, S=6, device=device)

training_dataset = get_training_dataset()
validation_dataset = get_validation_dataset()

################################################################################

delta_time = datetime.timedelta(hours=1)
timezone = datetime.timezone(offset=delta_time)

t = datetime.datetime.now(tz=timezone)
str_t = '{:%Y-%m-%d %H:%M:%S}'.format(t)
print(f"[START] : {str_t} :")
print(f"[Training on] : {str(device).upper()}")
print(f"Learning rate : {optimizer.defaults['lr']}")

EPOCHS = 10
S = 6


################################################################################

nb_it = []
batch_total_train_loss_list = []
batch_train_losses_list = []
batch_train_class_acc = []

batch_val_MSE_box_list = []
batch_val_confscore_list = []
batch_val_class_acc = []

for epoch in range(EPOCHS):
    update_lr(epoch, optimizer)

    begin_time = timer()
    epochs_loss = 0.
    
    print("-"*20)
    str_t = '{:%Y-%m-%d %H:%M:%S}'.format(t)
    print(" "*5 + f"{str_t} : EPOCH {epoch+1}/{EPOCHS}")
    print(" "*5 + f"Learning rate : lr = {optimizer.defaults['lr']}")
    print("-"*20)

    #########################################################################

    for batch, (img, bbox_true, labels) in tqdm_fct(training_dataset):
        model_MNIST.train()
        loss = 0
        begin_batch_time = timer()
        img, bbox_true, labels = img.to(device), bbox_true.to(device), labels.to(device)
        
        ### clear gradients
        optimizer.zero_grad()
        
        ### prediction (N,S,S,5) & (N,S,S,10)
        bbox_preds, label_preds = model_MNIST(img)
        
        ### compute losses over each grid cell for each image in the batch
        losses, loss = loss_yolo(bbox_preds, bbox_true, label_preds, labels)
    
        ### compute gradients
        loss.backward()
        
        ### Weight updates
        optimizer.step()

        ##### Class accuracy
        train_classes_acc = class_acc(bbox_true, labels, label_preds)

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

            pretty_print(batch, len(training_dataset.dataset), current_loss, losses, train_classes_acc)

            ############### Compute validation metrics each 100 batch ###########################################
            mse_box, mse_confidence_score, classes_acc = validation_loop(model_MNIST, validation_dataset, S, device)
            batch_val_MSE_box_list.append(mse_box)
            batch_val_confscore_list.append(mse_confidence_score)
            batch_val_class_acc.append(classes_acc)
            #####################################################################################################

            if batch == len(training_dataset.dataset)//BATCH_SIZE:
                print(f"Total elapsed time for training : {datetime.timedelta(seconds=timer()-begin_time)}")
                print(f"Mean training loss for this epoch : {epochs_loss / len(training_dataset):.5f}")
                print("\n\n")

path_save_model = f"yolo_mnist_model_{epoch}epochs_relativeCoords"
save_model(model_MNIST, path_save_model, SAVE_MODEL)