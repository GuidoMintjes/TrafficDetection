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

imageFile = 'zebra.jpg'
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



# bool recursed, String initString
def initProgramFirstCheck():
    checked = False
    initBool = False # Standaard nog niet geopend

    while not checked:
        # Vragen of het de eerste keer is van het programma en de checks ervoor snap je wel
        initString = input("Is dit de eerste keer dat je het yolov3 programma opent? (Y/N) ")
        f.br()

        if initString == "Y":

            initBool = True

        elif initString == "N":
            initBool = False
                
        else:
            print("Verkeerde input, start opnieuw...")
            f.br()
            continue
        
        checked = True
        
    return initBool


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
def getImageNStuff():
    
    f.br()
    f.br()

    # Load yolov3 model
    model = load_model(modelsFolder + '\\' + 'model.h5')

    image, image_w, image_h = f.load_image_pixels(imageFile, (input_w, input_h))

    # VVVVVV Niet nodig opzich

    #cv2.imshow("abc", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    image = expand_dims(image, 0) # De keras code verwacht 4 dimensies, 1 plaatje heeft 3 dimensies (dat kan je zien door print(image) te doen)
                                  # als je dat wilt in de regel hierboven.
                                  # De '0' in de expand_dims betekent dat er een nulde, oftwel een eerste (omdat arrays beginnen bij nul),
                                  # dimensie wordt toegevoegd --> de nulde, eerste en tweede dimensies van het plaatje schuiven op naar de
                                  # eerste, tweede en derde dimensie
                                  # Deze nieuwe nulde dimensie staat dan voor het aantal plaatjes, bij ons nu natuurlijk maar 1, maar keras
                                  # verwacht er vaak al meerdere tegelijk bij het model.predict stukje 



#----------------------------------------------------------------------------------------#

    # Maak de voorspelling met het model
    yhat = model.predict(image)

    # print([a.shape for a in yhat]) # Laat de dimensies zien van de array

    return image, image_w, image_h, model, yhat


def decodeFrame(image, image_w, image_h, model, yhat):
    
    boxes = list()
    for i in range(len(yhat)):
        # Decodeer de bounding boxes (de rechthoekjes om het herkende heen)
        boxes += dF.decode_netout(yhat[i][0], anchors[i], propability, input_h, input_w)

    # Zet de shape en size van de boxes weer om naar het originele plaatje ipv het 416x416  plaatje
    dF.correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)

    dF.do_nms(boxes, 0.5) # zorg ervoor dat overlappende bounding boxes weg gehaald worden( 0.5 staat voor bounding boxes die 50% over een komen)

    v_boxes, v_labels, v_scores = f.get_boxes(boxes, labels, propability)


    f.br()
    print("Die tensorflow error net:")
    print("'WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.'")
    print("Is niks ergs, maak geen zorgen")
    f.br()

    # eventjes de aparte boxen printen
    for i in range(len(v_boxes)):

        print(v_labels[i], v_scores[i])

    
    imageFinal = f.draw_boxes(imageFile, v_boxes, v_labels, v_scores, 0.5)
    
    cv2.imshow("window", imageFinal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(imageFile[:-4] + '_detected' + imageFile[-4:], (imageFinal).astype('uint8')) 



def __main__():
    
    downloadCheck()
    yoloModelCheck()
    
    f.br()
    print("De lokale weights en model bestanden kloppen, er kan verder gegaan worden met het herkennen...")
    f.br()

    f.br()
    print("Het (voorbeeld) plaatje (of de (voorbeeld) video) pakken en herkennen met het model!")
    image, image_w, image_h, model, yhat = getImageNStuff()

    decodeFrame(image, image_w, image_h, model, yhat)