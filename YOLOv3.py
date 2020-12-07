# imports
import os
import requests

import tensorflow as tf
from tensorflow import keras

import numpy as np

from cv2 import cv2

from functions import *


# Een paar standaard variabelen opstellen
preTrainedURL = 'https://pjreddie.com/media/files/yolov3.weights'

weightsFolder = 'weights'

# bool recursed, String initString
def initProgramFirstCheck():
    checked = False
    initBool = False # Standaard nog niet geopend

    while not checked:
        # Vragen of het de eerste keer is van het programma en de checks ervoor snap je wel
        initString = input("Is dit de eerste keer dat je het yolov3 programma opent? (Y/N) ")
        br()

        if initString == "Y":

            initBool = True

        elif initString == "N":
            initBool = False
                
        else:
            print("Verkeerde input, start opnieuw...")
            br()
            continue
        
        checked = True
        
    return initBool


# Downloaden kan er wat lastig uit zien, gebruikt het functions.py script 
def downloadCheck():
    bestandsPad = os.path.join(weightsFolder, "yolov3.weights")

    if os.path.exists(bestandsPad):
        lokaleGrootte = os.stat(bestandsPad).st_size
        externeGrootte = requests.get(preTrainedURL, stream=True).headers['content-length']

        if lokaleGrootte == externeGrootte:
            print("Het lokale weights bestand klopt! Ga verder...")

        else:
            print("Het lokale bestand klopt niet helemaal, hij wordt opnieuw gedownload...")
            download(preTrainedURL, weightsFolder, (1024 * 16))


    else:
        print('Programma is voor de eerste keer gestart, download wordt gestart...')
        download(preTrainedURL, weightsFolder, (1024 * 16))


# De eigenlijk start functie dat wordt gebruikt na de init funcs/defs
def start():
    
    

    return




def __main__():
    
    downloadCheck()

    start()