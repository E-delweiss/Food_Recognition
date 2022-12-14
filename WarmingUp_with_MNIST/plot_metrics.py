import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import confusion_matrix
import seaborn as sn

from smallnet import NetMNIST
from MNIST_dataset import get_validation_dataset
from validation import validation_loop

S=6
C=10
B=1

### Extracting losses
with open("WarmingUp_with_MNIST/results/train_results_10epochs.pkl", 'rb') as pkl:
    pickle_data_train = pickle.load(pkl)

with open("WarmingUp_with_MNIST/results/val_results_10epochs.pkl", 'rb') as pkl:
    pickle_data_val = pickle.load(pkl)

batch_train_losses_list = pickle_data_train["batch_train_losses_list"]
batch_train_class_acc = pickle_data_train["batch_train_class_acc"]
batch_val_losses_dict = pickle_data_val
#########################################################################

### Formatting training losses
xy_loss = []
wh_loss = []
confidence_withObject_loss = []
confidence_withoutObject_loss = [] 

for it in range(len(batch_train_losses_list)):
    xy_loss.append(batch_train_losses_list[it]['loss_xy'].detach().numpy())
    wh_loss.append(batch_train_losses_list[it]['loss_wh'].detach().numpy())
    confidence_withObject_loss.append(batch_train_losses_list[it]['loss_conf_obj'].detach().numpy())
    confidence_withoutObject_loss.append(batch_train_losses_list[it]['loss_conf_noobj'].detach().numpy())
#########################################################################

### Plot of box size losses
fig_loss_xywh = plt.figure()
plt.plot(xy_loss)
plt.plot(wh_loss)
plt.plot(batch_val_losses_dict["batch_val_MSE_box_list"])

plt.title("Training & val losses on bounding boxes")
plt.ylabel("MSE Loss")
plt.xticks([])
plt.ylim([0,0.5])
plt.legend(['xy_loss', 'wh_loss', 'MSE validation bounding box loss'])
plt.savefig('size_loss.png')
#########################################################################

### Plot of confidence score losses
fig_loss_confscore = plt.figure()
plt.plot(confidence_withObject_loss)
plt.plot(confidence_withoutObject_loss)
plt.plot(batch_val_losses_dict["batch_val_confscore_list"])

plt.title("Training & val losses on confidence scores")
plt.ylabel("MSE Loss")
plt.xticks([])
plt.ylim([0,1.5])
plt.legend(['C with object', 'C without object', 'MSE validation confidence score'])
plt.savefig('confidenceScore_loss.png')
#########################################################################

### Plot of accuracy
fig_training_acc = plt.figure()
train_class_acc = batch_train_class_acc
plt.plot(train_class_acc)
plt.plot(pickle_data_val["batch_val_class_acc"])
plt.title("Training and validation class accuracies")
plt.ylabel("Loglike")
plt.xticks([])
plt.ylim([0.2,1])
plt.legend(['train_class_acc', 'validation_class_acc'])
plt.savefig('accuracy.png')
#########################################################################

### Confusion matrix
fig_confusionMatrix = plt.figure(figsize = (12,7))
y_pred = []
y_true = []

### Loading model weights
model = NetMNIST(75, S=S, B=B, C=C)
model.load_state_dict(torch.load("WarmingUp_with_MNIST/results/MNIST_smallnet_10epochs_28102022_17h54.pt"))

### Validation loop
_, bbox_true, bbox_pred, labels, labels_pred = validation_loop(model, get_validation_dataset())

### keeping only cells (i,j) with an object 
cells_with_obj = bbox_true.nonzero()[::5]
N, cells_i, cells_j, _ = cells_with_obj.permute(1,0)
softmax_pred_classes = torch.softmax(labels_pred[N, cells_i, cells_j], dim=1)
argmax = torch.argmax(softmax_pred_classes, dim=1).cpu().numpy()
y_pred.extend(argmax)

labels = torch.argmax(labels, dim=1).cpu().numpy()
y_true.extend(labels)

# constant for classes
classes = [str(x) for x in range(C)]

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix)*C, index = [i for i in classes],
                     columns = [i for i in classes])

sn.heatmap(df_cm, annot=True)
plt.savefig('confusion_matrix.png')
