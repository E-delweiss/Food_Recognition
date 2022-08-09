# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 09:49:40 2022
"""

import glob
import numpy as np

from utils import plot_bboxes


path = r"/Users/thierryksstentini/Documents/Python_divers/GitHub/Food_Recognition/plato_dataset/obj_train_data"
data_txt = glob.glob(path + '/*.txt')
annotations = {}

for file in data_txt:
    obj_dict = {}
    with open(file, 'r') as f:
        line = f.readline()
        while line != '':
            annot_split = line.split()
            annot_split_float = [float(k) for k in annot_split[1:]]
            label = int(annot_split[0])
            obj_dict[label] = annot_split_float
            line = f.readline()
        
        key = file[file.rfind('/')+1:]
        annotations[key] = obj_dict
    



