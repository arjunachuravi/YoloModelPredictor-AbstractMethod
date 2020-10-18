# from modules import driverTestingFunction
# import cv2
# import numpy as np

# image,detections = driverTestingFunction(
#     'labelfile_test.txt',
#     'yolov4.cfg',
#     'yolov4.weights',
#     'test.jpg',
#     0.25,0.5,0.45
# )

# cv2.imwrite("filexyz.jpg",image)
# print(detections)

from train_test_prep import train_test_preparation_yolo

train_test_preparation_yolo(
    "","","CustomDataset\labelfile_train.txt"
)