import os, sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from configparser import ConfigParser

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sn

from yoloResnet import yoloResnet
from mealtrays_dataset import get_validation_dataset
from validation import validation_loop
from utils import get_cells_with_object

current_folder = os.path.dirname(locals().get("__file__"))
config_file = os.path.join(current_folder, "config.ini")
sys.path.append(config_file)

config = ConfigParser()
config.read('config.ini')
S = config.getint("MODEL", "GRID_SIZE")
C = config.getint("MODEL", "NB_CLASS")
B = config.getint("MODEL", "NB_BOX")

pkl_train_path = config.get("PICKLE", "pkl_train")
pkl_val_path = config.get("PICKLE", "pkl_val")

### Extracting losses
with open(pkl_train_path, 'rb') as pkl:
    pickle_data_train = pickle.load(pkl)

with open(pkl_val_path, 'rb') as pkl:
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
    wh_loss.append(batch_train_losses_list[it]['loss_wh'].detach().numpy() )
    confidence_withObject_loss.append(batch_train_losses_list[it]['loss_conf_obj'].detach().numpy() )
    confidence_withoutObject_loss.append(batch_train_losses_list[it]['loss_conf_noobj'].detach().numpy() )
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
model = yoloResnet(load_yoloweights=True, resnet_pretrained=False, S=S, C=C, B=B)

### Validation loop
img, target, prediction = validation_loop(model, get_validation_dataset())

### keeping only cells (i,j) with an object
N, cells_i, cells_j = get_cells_with_object(target)
softmax_pred_classes = torch.softmax(prediction[N, cells_i, cells_j, 2*5:], dim=1)
argmax = torch.argmax(softmax_pred_classes, dim=1).cpu().numpy()
y_pred.extend(argmax)

labels = torch.argmax(target[N, cells_i, cells_j, 5:], dim=1).cpu().numpy()
y_true.extend(labels)

# constant for classes
classes = [str(x) for x in range(8)]

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix)*C, index = [i for i in classes],
                     columns = [i for i in classes])

sn.heatmap(df_cm, annot=True)
plt.savefig('confusion_matrix.png')
