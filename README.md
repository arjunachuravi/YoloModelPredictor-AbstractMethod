Project Phase 1

## modules/driverTestingFunction

```
labelConfPath,  #--> labels with color
configPath,     #--> yolo config file
weightPath,     #--> yolo weight file
imagePath,      #--> test image path
MIN_CONF,
nms_conf, 
nms_thresh 
```
```
img , dec = driverTestingFunction(
    "labelfile_train.txt",
    "yolov4.cfg",
    "yolov4_last.weights",
    "6.jpeg",
    0.2,0.5,0.45
)  

cv2.imwrite("dest.jpeg",img)
print(dec)
```
