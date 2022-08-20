# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 22:07:40 2022
"""
import PIL
import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_bboxes(annotations, img_name, path):
    annotations_idx = annotations[img_name + '.txt']
    img_idx = PIL.Image.open(path + '/' + img_name + '.jpg').convert('RGB')
    img_W = img_idx.size[0]
    img_H = img_idx.size[1]

    fig, ax = plt.subplots()
    ax.imshow(img_idx)
    for k in annotations_idx.keys():
        cx = annotations_idx[k][0]
        cy = annotations_idx[k][1]
        rw = annotations_idx[k][2]
        rh = annotations_idx[k][3]
        
        cx_abs = img_W * cx
        cy_abs = img_H * cy
        
        x = cx_abs - (img_W * (rw/2)) 
        y = cy_abs - (img_H*(rh/2))
        
        w = img_W * rw
        h = img_H * rh
        
        color = {0:'r', 6:'b'}
        rect = patches.Rectangle((x, y), w, h, facecolor='none', edgecolor=color[k])
        ax.add_patch(rect)

    plt.show()




################################################################################
def display_digits_with_boxes(digits, predictions, labels, pred_bboxes, bboxes, iou, title):
  """Utility to display a row of digits with their predictions.

  Args:
    digits : np.ndarray of shape (N,75,75,1)
        Raw image with normalized pixel values (from 0 to 1)
    predictions : np.ndarray of shape (N,)
        Predicted label with the same shape as labels
    labels : np.ndarray of shape (N,)
        Labels of the digits (from 0 to 9)
    pred_bboxes : np.ndarray of shape (n, N) ??
        Predicted bboxes locations
    bboxes : np.ndarray of shape (n, N)
        Ground true bboxe locations
    iou : list of shape (???)
        IoU of each bboxes
    title : str
        Figure's title
  """
  n = 10
  indexes = np.random.choice(len(predictions), size=n)
  n_digits = digits[indexes]
  n_predictions = predictions[indexes]
  n_labels = labels[indexes]

  n_iou = []
  if len(iou) > 0:
    # If multiple bboxes
    n_iou = iou[indexes]

  if (len(pred_bboxes) > 0):
    # If multiple bboxes predicted
    n_pred_bboxes = pred_bboxes[indexes,:]

  if (len(bboxes) > 0):
    # If multiple ground truth bboxes
    n_bboxes = bboxes[indexes,:]

  # Rescale pixel values to un-normed values (from 0 -black- to 255 -white-)
  n_digits = n_digits * 255.0
  n_digits = n_digits.reshape(n, 75, 75)

  # Set plot config
  fig = plt.figure(figsize=(20, 4))
  plt.title(title)
  plt.yticks([])
  plt.xticks([])
  
  for i in range(10):
    ax = fig.add_subplot(1, 10, i+1)
    bboxes_to_plot = []
    if (len(pred_bboxes) > i):
      bboxes_to_plot.append(n_pred_bboxes[i])
    
    if (len(bboxes) > i):
      bboxes_to_plot.append(n_bboxes[i])

    img_to_draw = draw_bounding_boxes_on_image_array(image=n_digits[i], boxes=np.asarray(bboxes_to_plot), color=['red', 'green'], display_str_list=["true", "pred"])
    plt.xlabel(n_predictions[i])
    plt.xticks([])
    plt.yticks([])
    
    if n_predictions[i] != n_labels[i]:
      ax.xaxis.label.set_color('red')
    
    plt.imshow(img_to_draw)

    if len(iou) > i :
      color = "black"
      if (n_iou[i][0] < iou_threshold):
        color = "red"
      ax.text(0.2, -0.3, "iou: %s" %(n_iou[i][0]), color=color, transform=ax.transAxes)
################################################################################

################################################################################
def dataset_to_numpy_util(training_dataset, validation_dataset, N, S):
  """
  Pull a batch from the datasets. This code is not very nice.
  
  Args:
    training_dataset : torch.utils.data.Dataset
        Dataset from the torch.utils.data.Dataset Pytorch class, returning the 
        training digits, labels and bboxes coordinates as batches such as : 
            - training digits : torch.Tensor of shape (batch_size, 1, 75, 75)
            - labels : torch.Tensor of shape (batch_size, 10)
            - bboxes coordinates : torch.Tensor of shape (batch_size, 4)
    validation_dataset : torch.utils.data.Dataset 
        Dataset from the torch.utils.data.Dataset Pytorch class, returning the 
        whole validation digits, labels and bboxes coordinates.
    N : int
        Size of the training sample to extract from the training dataset
    S : int
        Number of cell in one direction

  Returns:
    N_train_ds_digits : np.ndarray of shape (N, 1, 75, 75)
    N_train_ds_labels : np.ndarray of shape (N, 10)
    N_train_ds_bboxes : np.ndarray of shape (N, 4)
    validation_digits : np.ndarray of shape (len(validation_dataset), 1, 75, 75)
    validation_labels : np.ndarray of shape (len(validation_dataset), 10)
    validation_bboxes : np.ndarray of shape (len(validation_dataset), 4)
  """
  ### get N training digits, labels and bboxes from one batch
  ### turning the bboxes coordinates into ndarrays
  one_batch_train_ds_digits, one_batch_train_ds_labels, one_batch_train_ds_bboxes = next(iter(training_dataset))
  cell_size = 75/6 #one_batch_train_ds_digits[0][0].shape[1]/ S

  N_train_ds_digits = one_batch_train_ds_digits[:N].numpy()
  N_train_ds_labels = one_batch_train_ds_labels[:N].numpy()
  
  N_train_ds_bboxes = one_batch_train_ds_bboxes[:N].numpy()
  N_train_ds_bboxes_abs = np.zeros((N,4))

  ####### A MODIFIER
  x_min = ((N_train_ds_bboxes[:,2])) - (N_train_ds_bboxes[:,4]/2)
  y_min = ((N_train_ds_bboxes[:,3])) - (N_train_ds_bboxes[:,5]/2)
  x_max = ((N_train_ds_bboxes[:,2])) + (N_train_ds_bboxes[:,4]/2)
  y_max = ((N_train_ds_bboxes[:,3])) + (N_train_ds_bboxes[:,5]/2)
  #######

  N_train_ds_bboxes_abs[:,0] = x_min
  N_train_ds_bboxes_abs[:,1] = y_min
  N_train_ds_bboxes_abs[:,2] = x_max
  N_train_ds_bboxes_abs[:,3] = y_max


  print(N_train_ds_bboxes_abs[5])
  
  ### get the whole validation digits, labels and bboxes
  ### turning the bboxes coordinates into ndarrays
  for validation_digits, validation_labels, validation_bboxes in validation_dataset:
      validation_digits = validation_digits.numpy()
      validation_labels = validation_labels.numpy()
      validation_bboxes = validation_bboxes.numpy()
      break

  # turning one hot encoding labels into the corresponding digit
  validation_labels = np.argmax(validation_labels, axis=1)
  N_train_ds_labels = np.argmax(N_train_ds_labels, axis=1)

  return (N_train_ds_digits, N_train_ds_labels, N_train_ds_bboxes_abs,
          validation_digits, validation_labels, validation_bboxes)




(training_digits, training_labels, training_bboxes,
 validation_digits, validation_labels, validation_bboxes) = dataset_to_numpy_util(training_dataset, validation_dataset, 10, S=6)
display_digits_with_boxes(training_digits, training_labels, training_labels, np.array([]), training_bboxes, np.array([]), "training digits and their labels")