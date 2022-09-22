import datetime
from timeit import default_timer as timer
import torch

from utils import device
from Yolo_loss import YoloLoss
from MNIST_dataset import get_training_dataset, get_validation_dataset
from Darknet_like import YoloMNIST
from Metrics import class_acc

learning_rate = 0.001
BATCH_SIZE = 64
device = device() 

model_MNIST = YoloMNIST(sizeHW=75, S=6, C=10, B=1)
model_MNIST = model_MNIST.to(device)
optimizer = torch.optim.Adam(params=model_MNIST.parameters(), lr=learning_rate, weight_decay=0.0005)
loss_yolo = YoloLoss(lambd_coord=5, lambd_noobj=0.5, S=6, device=device)

training_dataset, len_training_ds = get_training_dataset()
validation_dataset, len_validation_ds = get_validation_dataset()

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

for epoch in range(EPOCHS) :
    if epoch > 7:
        optimizer.defaults['lr'] = 0.0001

    begin_time = timer()
    epochs_loss = 0.
    
    print("-"*20)
    str_t = '{:%Y-%m-%d %H:%M:%S}'.format(t)
    print(" "*5 + f"{str_t} : EPOCH {epoch+1}/{EPOCHS}")
    print(" "*5 + f"Learning rate : lr = {optimizer.defaults['lr']}")
    print("-"*20)

    #########################################################################

    for batch, (img, bbox_true, labels) in enumerate(training_dataset):
        model_MNIST.train()
        loss = 0
        begin_batch_time = timer()
        img, bbox_true, labels = img.to(device), labels.to(device), bbox_true.to(device)
        
        ### clear gradients
        optimizer.zero_grad()
        
        ### compute predictions
        bbox_preds, label_preds = model_MNIST(img)
        
        ### compute losses over each grid cell for each image in the batch
        losses, loss = loss_yolo(bbox_preds, bbox_true, label_preds, labels)
    
        ### compute gradients
        loss.backward()
        
        ### Weight updates
        optimizer.step()

        ##### Class accuracy
        classes_acc = class_acc(bbox_true, labels, label_preds)

        ######### print part #######################
        current_loss = loss.item()
        epochs_loss += current_loss

        if batch+1 <= len_training_ds//BATCH_SIZE:
            current_training_sample = (batch+1)*BATCH_SIZE
        else:
            current_training_sample = batch*BATCH_SIZE + len_training_ds%BATCH_SIZE
        
        if (batch) == 0 or (batch+1)%100 == 0 or batch == len_training_ds//BATCH_SIZE:
            # Recording the total loss
            batch_total_train_loss_list.append(current_loss)
            # Recording each losses 
            batch_train_losses_list.append(losses)
            # Recording class accuracy 
            batch_train_class_acc.append(classes_acc)


            print(f"--- Image : {current_training_sample}/{len_training_ds}",\
                    f" : loss = {current_loss:.5f}")
            # print(f"xy_coord training loss for this batch : {torch.sum(losses['loss_xy']) / len(img):.5f}")
            print(f"xy_coord training loss for this batch : {losses['loss_xy']:.5f}")
            print(f"wh_sizes training loss for this batch : {losses['loss_wh']:.5f}")
            print(f"confidence with object training loss for this batch : {losses['loss_conf_obj']:.5f}")
            print(f"confidence without object training loss for this batch : {losses['loss_conf_noobj']:.5f}")
            print(f"class proba training loss for this batch : {losses['loss_class']:.5f}")
            print("\n")
            # print(f"Training class accuracy : {classes_acc.item()*100:.2f}%")

            # model_MNIST.eval()
            # for (img, labels, bbox_true) in validation_dataset:
            #     img, labels, bbox_true = img.to(device), labels.to(device), bbox_true.to(device)
            #     with torch.no_grad():
            #         ### prediction
            #         bbox_preds, label_preds = model_MNIST(img)

            #         ### (N,4) -> (N, S, S, 5)
            #         bbox_true_6x6, cells_i, cells_j = bbox2Tensor(bbox_true)
            #         bbox_true_6x6, cells_i, cells_j = bbox_true_6x6.to(device), cells_i.to(device), cells_j.to(device)
                    
            #         bbox_preds = bbox_preds.to(device)

            #         ### keeping only cells (i,j) with an object 
            #         # cells_with_obj = bbox_true_6x6.nonzero()[::5]
            #         # N, cells_i, cells_j, _ = cells_with_obj.permute(1,0)

            #         ### MSE along bbox coordinates and sizes in the cells containing an object
            #         N = range(len(img))
            #         mse_box = (1/len(img)) * torch.sum(torch.pow(bbox_true[:,:4] - bbox_preds[N, cells_i, cells_j,:4],2))
                    
            #         ### confidence score accuracy : sum of the all grid confidence scores
            #         ### pred confidence score is confidence score times IoU.
            #         mse_confidence_score = torch.zeros(len(img)).to(device)
            #         for i in range(S):
            #             for j in range(S):
            #                 iou = intersection_over_union(bbox_true_6x6[:,i,j], bbox_preds[:,i,j], cells_i, cells_j).to(device)
            #                 mse_confidence_score += torch.pow(bbox_true_6x6[:,i,j,-1] - bbox_preds[:,i,j,-1] * iou,2)
                    
            #         mse_confidence_score = (1/(len(img))) * torch.sum(mse_confidence_score)

            #         ### applied softmax to class predictions and compute accuracy
            #         softmax_pred_classes = torch.softmax(label_preds[N, cells_i, cells_j], dim=1)
            #         classes_acc = (1/len(img)) * torch.sum(torch.argmax(labels, dim=1) == torch.argmax(softmax_pred_classes, dim=1))


            #         batch_val_MSE_box_list.append(mse_box.item())
            #         batch_val_confscore_list.append(mse_confidence_score.item())
            #         batch_val_class_acc.append(classes_acc.item())

                    # print("|")
                    # print(f"| MSE validation box loss : {mse_box.item():.5f}")
                    # print(f"| MSE validation confidence score : {mse_confidence_score.item():.5f}")
                    # print(f"| Validation class acc : {classes_acc.item()*100:.2f}%")
                    # print("\n")

            if batch == (len_training_ds//BATCH_SIZE):
                print(f"Total elapsed time for training : {datetime.timedelta(seconds=timer()-begin_time)}")
                print(f"Mean training loss for this epoch : {epochs_loss / len(training_dataset):.5f}")
                print("\n\n")

    

