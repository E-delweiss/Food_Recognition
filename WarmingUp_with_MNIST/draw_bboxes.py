import torch
import numpy as np
import matplotlib.pyplot as plt 
import os
import PIL

import IoU

############### Matplotlib config
plt.rc('image', cmap='gray')
plt.rc('grid', linewidth=0)
plt.rc('xtick', top=False, bottom=False, labelsize='large')
plt.rc('ytick', left=False, right=False, labelsize='large')
plt.rc('axes', facecolor='F8F8F8', titlesize="large", edgecolor='white')
plt.rc('text', color='a8151a')
plt.rc('figure', facecolor='F0F0F0')# Matplotlib fonts
MATPLOTLIB_FONT_DIR = os.path.join(os.path.dirname(plt.__file__), "mpl-data/fonts/ttf")
################################################################################

def temp(img, label_true, label_pred, box_true, box_pred, titre, nb_sample=10):
    """
    TBM
    /!\ box_pred i.e. box_pred_NMS
    """
    indexes = np.random.choice(len(img), size=nb_sample)

    ### Choose image & rescale pixel values to un-normed values (from 0 -black- to 255 -white-)
    n_imgs = n_imgs[indexes].numpy()
    n_imgs = n_imgs * 255.0
    n_imgs = n_imgs.reshape(nb_sample, 75, 75)

    ### Choose label & argmax of one-hot vectors. 
    n_label_pred = label_pred[indexes]
    n_label_pred = torch.argmax(torch.softmax(n_label_pred, dim=-1), dim=-1).numpy() #(N,S,S,10) -> (N,S,S)

    ### Choose box_pred
    n_box_pred = box_pred[indexes]

    ### TBM
    n_box_true = box_true[indexes]
    n_bboxes = IoU.relative2absolute().numpy()
