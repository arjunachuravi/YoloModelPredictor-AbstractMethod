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
    "labelfile_train.txt",      #-->eg:jacket 255,230,120 and next line , next entry .. and so on
    "yolov4.cfg",               #-->yolo config 
    "yolov4_last.weights",      #-->yolo weight with structure corresponding to that of config (u can change img dims if you want in the code)
    "6.jpeg",                   #-->image to be predicted
    0.2,0.5,0.45                #-->min confidence(to consider from a pool of bounding boxes),Confidence,tuning thesholds
)  

cv2.imwrite("dest.jpeg",img)
print(dec)
```
