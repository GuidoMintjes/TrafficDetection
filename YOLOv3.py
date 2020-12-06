# imports
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

        initString = input("Is dit de eerste keer dat je het yolov3 programma opent? (Y/N)")
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



def start():
    print("a")




def __main__():
    antwoord = initProgramFirstCheck()
    
    if antwoord:
        download(preTrainedURL, weightsFolder)
        print("Gedownload!")
    
    else:
        #Hoeft niet te downloade
        print('b')

    start()