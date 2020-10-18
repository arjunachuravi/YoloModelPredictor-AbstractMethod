import cv2
import numpy as np

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
                    values = (splited[1].split(","))
                    values = [int(item) for item in values]
                    self.labels_colors[splited[0]] = values
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

    def inputImgDim(self):
        return self.DIMS

class Box:

    def __init__(self,boundaryBox,imageDims):
        self.boundaryBox = boundaryBox * np.array(
            [imageDims[1],imageDims[0],imageDims[1],imageDims[0]]
        )
        self.boundaryBox[:] = self.boundaryBox.astype("int")

    def getBoxDims(self):
        # co ordinate start_x,start_y,w,h
        return (
            int(self.boundaryBox[0] - (self.boundaryBox[2] / 2)),
            int(self.boundaryBox[1] - (self.boundaryBox[3] / 2)),
            self.boundaryBox[2],
            self.boundaryBox[3]
        )
    
    def generateEnds(self):
        return (
            int(self.boundaryBox[0] + self.boundaryBox[2]),
            int(self.boundaryBox[1] + self.boundaryBox[3])
        )
    
def driverTestingFunction(
    labelConfPath,configPath,weightPath,imagePath,MIN_CONF,nms_conf, nms_thresh ):

    detections = []
    conf = Configs(labelConfPath)
    labels = conf.getKeys()

    model = YoloModelPrediction(configPath,weightPath,imagePath)

    class_ids_list = []
    boxes_list = []
    confidences_list = []

    for detection_layer in model.obj_det_layers():
        for obj_feature in detection_layer:
            scores = obj_feature[5:]
            c_id = np.argmax(scores)
            confidence = scores[c_id]
            del(scores)
            
            if confidence > MIN_CONF:
                # sync-data-struct
                boxes_list.append(Box(obj_feature[:4],model.inputImgDim()))
                confidences_list.append(float(confidence))
                class_ids_list.append(c_id)
    
    max_value_ids = cv2.dnn.NMSBoxes(
        [list(item.getBoxDims()) for item in boxes_list],
        confidences_list, nms_conf, nms_thresh
    )

    # the cv2 fn will perform nms and get index of box with greatest confidence per class
    for max_valueid in max_value_ids:

        greatest_confidence_id = max_valueid[0]
        start_x_pt ,start_y_pt ,box_width ,box_height = boxes_list[greatest_confidence_id].getBoxDims()
        end_x_pt ,end_y_pt = boxes_list[greatest_confidence_id].generateEnds()

        predicted_class_id = class_ids_list[greatest_confidence_id]
        box_color = conf.getColor(labels[predicted_class_id])
        
        predicted_class_label = "{}: {:.2f}%".format(labels[predicted_class_id], confidences_list[greatest_confidence_id] * 100)
        detections.append((labels[predicted_class_id], confidences_list[greatest_confidence_id]))

        cv2.rectangle(model.IMAGE, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color,1)
        cv2.putText(model.IMAGE, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
    
    return model.IMAGE,detections