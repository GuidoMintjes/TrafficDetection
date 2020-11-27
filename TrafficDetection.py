# Import de coco tools voor de coco dataset ==> (cocodataset.org)
from pycocotools.coco import COCO

import sys

import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.io as io


print("De COCO 2017 dataset wordt automatisch gedownload, de bosch dataset moet handmatig gedownload worden")
answer = input('Wil je de COCO dataset of de yolov3 bosch dataset gebruiken voor het trainen? (C/Y)')


if answer == 'C':
    # Open het COCO script
    print('a')


elif answer == 'Y':
    # Functie om te vragen of dit de eerste keer is dat het programma wordt opgestart, voor nu nog niet echt gebruikt!

    # bool recursed, String initString
    def initProgramFirstCheck():
        checked = False
        initBool = False # Standaard nog niet geopend

        while not checked:

            initString = input("Is dit de eerste keer dat je het yolov3 programma opent? (Y/N)")
            
            if initString == "Y":
                initBool = True

            elif initString == "N":
                initBool = False
                
            else:
                print("Verkeerde input, start opnieuw...")
                continue
        
            checked = True
        
        return initBool


    x = initProgramFirstCheck()
    print(x)


else:
    print("Verkeerde input gekregen, probeer het alstublieft opnieuw!")
    sys.exit()