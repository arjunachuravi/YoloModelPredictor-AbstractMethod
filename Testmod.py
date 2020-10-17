from modules import driverFunction
import cv2
import numpy as np

image,detections = driverFunction(
    'labelfile.txt',
    'yolov4.cfg',
    'yolov4.weights',
    'test.jpg',
    0.25,0.5,0.45
)

cv2.imwrite("filexyz.jpg",image)
print(detections)