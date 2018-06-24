import os
import sys

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
from PIL import Image
from numpy import newaxis
import sys
from ssd import build_ssd
from matplotlib import pyplot as plt
from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
from data import VOC_CLASSES as labels
from Action_Recognition.action_recognizer import ActionRecognizer
import threading
import time

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

sys.path.insert(0, "/home/abu/Documents/Personal/ImageIdentifier/Frame-Prediction/Object Detection/Action_Recognition/")

class ParallelThread(threading.Thread):
    
    flag = True
    
    def __init__(self, callback):
        self.callback = callback
        threading.Thread.__init__(self)

    def run(self):
        while self.flag:
            self.callback()
            time.sleep(1)
    
    def stop(self):
        self.flag = False


 
class ObjectDetector:

    def __init__(self):
        print("Object Detector initialized")
        self.identifier_last_update_frame_map = dict()
        self.identifier_coordinate_map = dict()
        self.identifier_action_map = dict()
        self.identifier_image_queue_map = dict()
        self.action_recognizer = ActionRecognizer()
        self.action_recognizer_execution_thread = ParallelThread(self.getComputedAction)
        self.action_recognizer_execution_thread.start()
 
    

    def generateUniqueIdentifier(self, coordinates):
        return ((coordinates[0][0]+coordinates[0][1])/(coordinates[1]+coordinates[2]))

    def getSubImage(self, coords, image):
        images = []
        for i in range(0,375):
            if i>coords[0][1] and i<(coords[0][1]+coords[2]):
                row = []
                for j in range(0,500):
                        if j>coords[0][0] and j<(coords[1]+coords[0][0]):
                            row.append(image[i][j])
                images.append(row)
        images = np.asarray(images)
        return images


    def saveImage(self, image):
        image = Image.fromarray(image, 'RGB')
        image.save('filler.png')
        image.show()

    def validDifference(self, cord, coordinates, earlier_coordinates):
        limit = 225
        if abs(coordinates[0][cord]-earlier_coordinates[0][cord])<limit and abs(coordinates[cord+1]-earlier_coordinates[cord+1])< limit:
            return True
        return False

    def getMappedIdentifier(self, coordinates):
        for key in self.identifier_coordinate_map.keys():
            earlier_coordinates = self.identifier_coordinate_map[key]
            if self.validDifference(0, coordinates, earlier_coordinates) and self.validDifference(1, coordinates, earlier_coordinates):
                self.identifier_coordinate_map[key]= coordinates
                return key
        return None

    def getComputedAction(self):
        for u_id in self.identifier_image_queue_map:
            if (len(self.identifier_image_queue_map[u_id])>4):
                detected_action = self.action_recognizer.identifyAction(self.identifier_image_queue_map[u_id])
                self.identifier_action_map[u_id] = detected_action
                self.identifier_image_queue_map[u_id] = [];
            else:
                print("Waiting for data!!")

    def formattedImage(self, image):
        image = image.transpose(2,0,1)
        leftPad = max(round(float((112 - image.shape[1])) / 2),0)
        rightPad = max(round(float(112 - image.shape[1]) - leftPad),0)
        topPad = max(round(float((112 - image.shape[2])) / 2),0)
        bottomPad = max(round(float(112 - image.shape[2]) - topPad),0)
        pads = ((leftPad,rightPad),(topPad,bottomPad))
        img_arr = np.ndarray((3,112,112),np.int)
        image = image[:,:112,:112]
        print(image.shape)
        for i,x in enumerate(image):
            cons = np.int(np.median(x))
            x_p = np.pad(x,pads,'constant',constant_values=0)
            img_arr[i,:,:] = x_p
        
        return Image.fromarray(np.uint8(img_arr).transpose(1,2,0))


    def detect_objects(self, image, frame_count):

        net = build_ssd('test', 300, 21)    # initialize SSD
        net.load_weights('../weights/ssd300_mAP_77.43_v2.pth')
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        x = cv2.resize(image, (300, 300)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)

        xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
        if torch.cuda.is_available():
            xx = xx.cuda()
        y = net(xx)


        # plt.figure(figsize=(10,10))
        # colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        #plt.imshow(rgb_image)  # plot the image for matplotlib
        #currentAxis = plt.gca()

        detections = y.data

        # scale each detection back up to the image
        scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
        detected = {}
        for i in range(detections.size(1)):
            j = 0
            while detections[0,i,j,0] >= 0.6:
                #score = detections[0,i,j,0]
                label_name = labels[i-1]
                #display_txt = '%s: %.2f'%(label_name, score)
                pt = (detections[0,i,j,1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                detected.setdefault(label_name,[]).append([self.getSubImage(coords, image),coords])
                # color = colors[i]
                # currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                # currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
                j+=1

        for key in detected:
            #We are just focused on uniquely identifying people as only human action is recognized now.
            if key != "person":
                continue
            for entry in detected[key]:
                u_id = self.getMappedIdentifier(entry[1])
                if u_id==None:
                    u_id = self.generateUniqueIdentifier(entry[1])
                    self.identifier_coordinate_map[u_id] = entry[1]
                self.identifier_image_queue_map.setdefault(u_id,[]).append(self.formattedImage(entry[0]))
                self.identifier_last_update_frame_map[u_id] = frame_count



c = ObjectDetector()
cap = cv2.VideoCapture('video1.mp4')
i=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
while(True and i>0):
    ret, frame = cap.read()
    if i%5==0:
        c.detect_objects(np.asarray(frame),i)
    i-=1

    
cap.release()
