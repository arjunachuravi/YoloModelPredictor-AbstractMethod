import numpy as np
import cv2

class Image:

    def __init__(self,imagePath):
        self.PATH = imagePath
        self.IMAGE = cv2.imread(self.PATH)
        # H W 
        self.DIMS = (
            self.IMAGE.shape[0],self.IMAGE.shape[1]
        )
        self.SCALE = 0.003922 # 1/255
    
    def imageBlob(self):
        return cv2.dnn.blobFromImage(
            self.IMAGE,
            self.SCALE,
            (416,416),
            swapRB = True,
            crop = False
        )
    
class Configs:

    def __init__(self,fileName):

        self.labels_colors = {}

        with open(fileName) as target:
            labels_colors_list = target.readlines()
            labels_colors_list[:] = [
                item.strip("\n") for item in labels_colors_list
            ]
            for item in labels_colors_list:
                splited = item.split()
                if not splited[0] in self.labels_colors.keys():
                    self.labels_colors[splited[0]] = (splited[1].split(",")).astype("int")
            del(labels_colors_list)

    def getKeys(self):
        return list(self.labels_colors.keys())

    def getColor(self,key):
        return self.labels_colors[key] if key in self.getKeys() else [255,255,255]

class YoloModelPrediction(Image):

    def __init__(self,configPath,weightPath,imagePath):
        super().__init__(imagePath)
        self.yolo_model = cv2.dnn.readNetFromDarknet(
            configPath, weightPath
        )
        self.yolo_layers = self.yolo_model.getLayerNames()
        self.yolo_output_layer = [
            self.yolo_layers[yolo_layer[0] - 1] for yolo_layer in self.yolo_model.getUnconnectedOutLayers()
        ]
        self.yolo_model.setInput(self.imageBlob())

    def obj_det_layers(self):
        return self.yolo_model.forward(self.yolo_output_layer)

class Box:

    def __init__(self,boundaryBox,imageDims):
        self.boundaryBox = boundaryBox * np.array(
            [imageDims[1],imageDims[0],imageDims[1],imageDims[0]]
        )
        self.boundaryBox[:] = self.boundaryBox.astype("int")

    def getBoxDims(self):
        # co ordinate start_x,start_y,w,h
        return [
            int(self.boundaryBox[0] - (self.boundaryBox[2] / 2)),
            int(self.boundaryBox[1] - (self.boundaryBox[3] / 2)),
            self.boundaryBox[2],
            self.boundaryBox[3]
        ]
    
def driverFunction(
    labelConfPath,configPath,weightPath,imagePath,MIN_CONF,nms_conf, nms_thresh
):
    
    # conf = Configs(labelConfPath)
    # labels = conf.getKeys()
    labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
                "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
                "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
                "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
                "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
                "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
                "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
                "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
                "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
                "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]


    model = YoloModelPrediction(configPath,weightPath,imagePath)
    detection_layers = model.obj_det_layers()

    class_ids_list = []
    boxes_list = []
    confidences_list = []

    for detection_layer in detection_layers:
        for obj_feature in detection_layer:
            scores = obj_feature[5:]
            c_id = np.argmax(scores)
            confidence = scores[c_id]

            if confidence > MIN_CONF:
                box = Box(obj_feature[:4],model.DIMS)
                boxes_list.append(box.getBoxDims())
                del(box)
                confidences_list.append(float(confidence))
                class_ids_list.append(c_id)
    
    max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, nms_conf, nms_thresh)
    image = cv2.imread(imagePath)

    for max_valueid in max_value_ids:

        max_class_id = max_valueid[0]

        box = boxes_list[max_class_id]
        start_x_pt = box[0]
        start_y_pt = box[1]
        box_width = box[2]
        box_height = box[3]

        predicted_class_id = class_ids_list[max_class_id]
        predicted_class_label = labels[predicted_class_id]
        prediction_confidence = confidences_list[max_class_id]
        # box_color = conf.getColor(predicted_class_label)

        end_x_pt = int(start_x_pt + box_width)
        end_y_pt = int(start_y_pt + box_height)


        predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
        print("predicted object {}".format(predicted_class_label))

        cv2.rectangle(image, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), [250,30,0],1)
        cv2.putText(image, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [250,30,0], 1)
    
    cv2.imwrite("Detection.jpg", image)

driverFunction(
    "","yolov4.cfg","yolov4.weights","test.jpg",0.4,0.5,0.4
)