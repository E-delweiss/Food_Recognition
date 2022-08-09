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