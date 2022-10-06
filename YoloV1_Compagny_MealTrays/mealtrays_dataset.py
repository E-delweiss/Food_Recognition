import glob
import numpy as np
import random as rd
import os

import PIL
import torch
import torch.nn.functional as F
import torchvision

class my_mnist_dataset(torch.utils.data.Dataset):
    def __init__(self, root:str, split:str="train", download:bool=False, S=7):
        self.root = root
        self.label_names = {0:'Assiette', 1:'Entree', 2:'Pain', 3:'Boisson', 4:'Yaourt', 5:'Dessert', 6:'Fruit', 7:'Fromage'}
        self.data_txt = glob.glob(root + '/*.txt')
        self.data_img = glob.glob(root + '/*.jpg')

    def build_annotations(self, data_txt, data_img):
        """
        Example : 
        For key (i.e. image) : '20220308112654_000042_059_0000000001_B______xx_C (1636)'
        Output = [
            ['Assiette', 0, 0.414109, 0.291401, 0.412906, 0.565905],
            ['Dessert', 5, 0.233531, 0.713405, 0.1885, 0.362759],
            ['Entree', 1, 0.510547, 0.746034, 0.319969, 0.411121],
            ['Fruit', 6, 0.742875, 0.764784, 0.158688, 0.243017],
            ['Boisson', 3, 0.673172, 0.466121, 0.133281, 0.230948]
            ]
        """
        self.annotations = {}
        for file in data_txt:
            obj_list = []
            with open(file, 'r') as f:
                line = ['_']
                while len(line) != 0:
                    line = f.readline()
                    line_list = [float(k) for k in line_list.split()]
                    label = int(line_list[0])
                    line_list[0] = label
                    line_list.insert(0, self.label_names.get(label))
                    
                    obj_list.append(line_list)
                    
                if obj_list:
                    start = file.rfind('/')+1
                    end = file.rfind('.txt')
                    key = file[start : end]
                    self.annotations[key] = obj_list

        def resize(self, annotations):
            for img_name, list_objects in annotations.items():
                img_PIL = PIL.Image.open(os.path.join(self.root, img_name + 'jpg')).convert('RGB')
                img_W = img_PIL.size[0]
                img_H = img_PIL.size[1]

                for object in list_objects:
                    xcr = object[2]
                    ycr = object[3]
                    wr = object[4]
                    hr = object[5]

                    xc_abs = img_W * xcr
                    yc_abs = img_H * ycr
                    xmin = xc_abs - (img_W * (wr/2)) 
                    ymin = yc_abs - (img_H * (hr/2))
