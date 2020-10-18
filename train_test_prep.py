import os
import shutil
from os import listdir
from os.path import isfile, join
from pathlib import Path as path
from sklearn.model_selection import train_test_split

def train_test_preparation_yolo(
    training_img_data_path,training_label_data_path,label_file ):

    train_file_gen_path         = "training.txt"
    test_file_gen_path          = "testing.txt"
    names_file                  = "labels.names"
    data_file                   = "cov.data"

    image_files = [f for f in listdir(training_img_data_path) if isfile(join(training_img_data_path, f))]
    labels_file = [f for f in listdir(training_label_data_path) if isfile(join(training_label_data_path, f))]

    with open(train_file_gen_path,"w") as file :
        for item in image_files:
            file.write(item + "\n")

    with open(test_file_gen_path,"w") as file :
        for item in image_files:
            file.write(item + "\n")
            
    f = open(names_file, "w")
    count = 0
    with open(label_file) as file:
        for line in file:
            count+=1
            line = line.split(" ")
            f.write(line[0]+"\n")
    f.close()

    with open(data_file,"w") as file :
        file.write(f"classes = {count}\n")
        file.write(f"training = {train_file_gen_path}\n")
        file.write(f"testing = {test_file_gen_path}\n")
        file.write(f"names = {names_file}\n")
        file.write(f"backup = backup")

    for file in labels_file:
        shutil.copy(os.path.join(training_label_data_path,file), training_img_data_path)
