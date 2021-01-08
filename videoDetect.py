import YOLOv3
from cv2 import cv2

import functions as f
import DetectionFunctions as dF

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from tensorflow.keras.models import load_model

import numpy as np
from numpy import expand_dims

import YOLOv3
import videoDetectTest


weightsFolder = 'weights'
modelsFolder = 'models'

input_w, input_h = (416, 416)

propability = 0.6
anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]

labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]




def decodeFrame(image, image_w, image_h, model, yhat, imageRead):
    
    boxes = list()
    for i in range(len(yhat)):

        boxes += dF.decode_netout(yhat[i][0], anchors[i], propability, input_h, input_w)


    dF.correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)

    dF.do_nms(boxes, 0.5) 
    
    v_boxes, v_labels, v_scores = f.get_boxes(boxes, labels, propability)

    imageFinal = f.draw_boxes(imageRead, v_boxes, v_labels, v_scores, 0.5)
    return imageFinal



def getImageNStuff(imageRead, vidWidth, vidHeight, model):
    
    imageRead = cv2.cvtColor(imageRead, cv2.COLOR_BGR2RGB)


    imageRead, image_w, image_h = f.load_image_pixels_video(imageRead, vidWidth, vidHeight)



    imageRead = expand_dims(imageRead, 0) 

    print(imageRead.shape)


    yhat = model.predict(imageRead)

    return imageRead, image_w, image_h, model, yhat, imageRead


def __main__():
    

    model = videoDetectTest.loadModel()

    videoFile = input("Vul hier de bestandsnaam van de te detecteren video in (zorg dat ie in dezelfde map zit!): ")
    
    if(videoFile == '0'):
        videoFile = 0
    
    video_capture = cv2.VideoCapture(videoFile)

    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    vidWidth = video_capture.get(3)
    vidHeight = video_capture.get(4)
    fps = video_capture.get(5)
    print(vidWidth, vidHeight, fps)

    if(videoFile != '0'):
        out = cv2.VideoWriter(videoFile[:-4] + '_detected' + videoFile[-4:],
                            fourcc,
                            fps,
                            (int(vidWidth), int(vidHeight)))

    videoDetectTest.downloadCheck()
    videoDetectTest.yoloModelCheck()

    frameNo = 1
    
    if(videoFile != '0'):
        totalFrames = int(cv2.VideoCapture.get(video_capture, int(cv2.CAP_PROP_FRAME_COUNT)))

        print(totalFrames)

    while(True):

        ret, frame = video_capture.read()
        
        print(frame)

        imageFinal = videoDetectTest.__main__(model, frame)

        cv2.imshow('Video', imageFinal)
        
        if(videoFile != '0'):
            out.write(cv2.resize(imageFinal, (int(vidWidth), int(vidHeight))))

            print("Frame " + str(frameNo) + " van de " + str(totalFrames) + " frames in de video is opgeslagen!")
        frameNo += 1

        if cv2.waitKey(1) & 0xFF == ord('Q'):
            print("Gestopt!")

            break
    
    video_capture.release()
    cv2.destroyAllWindows()

    out.release()