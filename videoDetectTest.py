# imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import requests

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import numpy as np
from numpy import expand_dims

from cv2 import cv2
from matplotlib import pyplot

import functions as f

import DetectionFunctions as dF

from cv2 import cv2




# Een paar standaard variabelen opstellen
preTrainedURL = 'https://pjreddie.com/media/files/yolov3.weights'

weightsFolder = 'weights'
modelsFolder = 'models'

# Voor de zekerheid even een leeg model variabele maken
model = None

propability = 0.6
anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]

imageFile = "zebra.jpg"
input_w, input_h = (416, 416)


# dit zijn alle dingen die we willen herkennen, kijk op cocodataset.org om te zien welke er allemaal zijn
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



def loadModel():
    model = load_model(modelsFolder + '\\' + 'model.h5')
    return model




# Downloaden kan er wat lastig uit zien, gebruikt het functions.py script 
def downloadCheck():

    bestandsPad = os.path.join(weightsFolder, "yolov3.weights")

    if os.path.exists(bestandsPad):
        lokaleGrootte = int(os.stat(bestandsPad).st_size)
        externeGrootte = int(requests.get(preTrainedURL, stream=True).headers['content-length'])

        
        if lokaleGrootte == externeGrootte:
            print("Het lokale weights bestand klopt! Ga verder...")
            f.br()

        else:
            print("Het lokale bestand klopt niet helemaal, hij wordt opnieuw gedownload...")
            f.download(preTrainedURL, weightsFolder, (1024 * 16))


    else:
        print('Programma is voor de eerste keer gestart, download wordt gestart...')
        f.download(preTrainedURL, weightsFolder, (1024 * 16))


def yoloModelCheck():

    f.folderCheck(modelsFolder) # Check of de folder voor het model al wel bestaat

    bestandsPad = os.path.join(modelsFolder, "model.h5")

    if os.path.exists(bestandsPad):
        return

    else:

        # Maak het model van de gedownloade weights
        model = dF.make_yolov3_model()


        weight_reader = dF.WeightReader(r"weights/yolov3.weights")


        weight_reader.load_weights(model)


        model.save(modelsFolder + '\\' + 'model.h5')
        
        return


# De eigenlijk start functie dat wordt gebruikt na de init funcs/defs
def getImageNStuff(imagee, model):


    image, image_w, image_h = f.load_image_pixels_video(imagee, input_w, input_h)


    image = expand_dims(image, 0) 


    # Maak de voorspelling met het model
    yhat = model.predict(image)

    return image, image_w, image_h, model, yhat


def decodeFrame(image, image_w, image_h, model, yhat, frame):
    
    boxes = list()
    for i in range(len(yhat)):
        # Decodeer de bounding boxes (de rechthoekjes om het herkende heen)
        boxes += dF.decode_netout(yhat[i][0], anchors[i], propability, input_h, input_w)

    # Zet de shape en size van de boxes weer om naar het originele plaatje ipv het 416x416  plaatje
    dF.correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)

    dF.do_nms(boxes, 0.5) # zorg ervoor dat overlappende bounding boxes weg gehaald worden( 0.5 staat voor bounding boxes die 50% over een komen)

    v_boxes, v_labels, v_scores = f.get_boxes(boxes, labels, propability)




    
    imageFinal = f.draw_boxes_video(frame, v_boxes, v_labels, v_scores, 0.5)


    return imageFinal



def __main__(model, frame):
    

    image, image_w, image_h, model, yhat = getImageNStuff(frame, model)

    imageFinal = decodeFrame(image, image_w, image_h, model, yhat, frame)

    return imageFinal