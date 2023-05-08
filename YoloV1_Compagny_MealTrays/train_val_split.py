import glob
import os
import shutil
from sklearn.model_selection import train_test_split


dataset = glob.glob("../../mealtray_dataset/dataset/obj_train_data/*.txt")
dataset = [txt_file for txt_file in dataset if os.path.getsize(txt_file) > 0]
dummy_target = range(len(dataset))
X_train, X_test, y_train, y_test = train_test_split(dataset, dummy_target, test_size=0.33)
print(len(X_train), len(X_test))

for txt in X_train:
    shutil.copy(txt, "../data/mealtray_dataset/train"+"/"+txt.split("/")[-1])
    shutil.copy(txt.replace(".txt", ".jpg"), "../data/mealtray_dataset/train"+"/"+txt.split("/")[-1].replace(".txt", ".jpg"))

for txt in X_test:
    shutil.copy(txt, "../data/mealtray_dataset/val"+"/"+txt.split("/")[-1])
    shutil.copy(txt.replace(".txt", ".jpg"), "../data/mealtray_dataset/val"+"/"+txt.split("/")[-1].replace(".txt", ".jpg"))