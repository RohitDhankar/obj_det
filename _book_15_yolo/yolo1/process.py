### SOURCE >> https://github.com/theAIGuysCode/yolov4-deepsort

import glob
import os
import numpy as np
import sys

current_dir = "darknet/data/own_data_dir" #
split_pct = 10#;
file_train = open("train.txt", "w")  
file_val = open("test.txt", "w")  
counter = 1  
index_test = round(100 / split_pct)  
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):  
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        if counter == index_test:
                counter = 1
                file_val.write(current_dir + "/" + title + '.jpg' + "\n")
        else:
                file_train.write(current_dir + "/" + title + '.jpg' + "\n")
                counter = counter + 1
file_train.close()
file_val.close()

#!./darknet detector train data/own_data_dir.data cfg/yolo4-custom.cfg yolo4.conv.137 -dont_show -map
#detector -dont_show -map train data/obj.data cfg/yolo-obj.cfg data/yolov4.conv.137
#sudo chmod -R 777 /var/www
# Couldnt open file data/multiple_images.data

# Couldnt open file: cfg/yolo4-custom.cfg

# !pwd
# %cd /content/drive/MyDrive/rohit_root_yolo/darknet 
# !pwd
# %cd /content/drive/MyDrive/rohit_root_yolo/
# !pwd
# !chmod -R 777 ./darknet/
# !./darknet detector train data/own_data_dir.data cfg/yolo4_custom.cfg yolo4.conv.137 -dont_show -map