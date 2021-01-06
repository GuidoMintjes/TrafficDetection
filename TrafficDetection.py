# Import de coco tools voor de coco dataset ==> (cocodataset.org)


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import importlib
from pathlib import Path

import functions as f

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io

# Eigen python packages
import YOLOv3
import videoDetect

# import TrafficDetection

# Start met printen naar de console en dingen vragen en dergelijke
f.br()


def __main__():
    print("Voordat de AI in gebruik kan worden genomen zal er automatisch een getraind model gedownload moeten worden...")
    print("Om dit helemaal succesvol te doen, moet bij het eerste gebruik bij de volgende vraag het plaatje worden geselecteerd...")
    print("Wanneer er wordt gevraagd om de bestandsnaam van een plaatje, kan het voorbeeldplaatje 'zebra.jpg' gebruikt worden...")
    f.br()

    answer = input('Wil je de AI op een (P)laatje testen, of op een (V)ideo? (P/V) ')
    f.br()

    #if answer == 'C':
    #    # Open het COCO script
    #    COCO.__main__()


    if answer == 'P':
        # Open het YOLO script
        YOLOv3.__main__()

    elif answer == 'V':

        videoDetect.__main__()
        sys.exit()


    else:
        answerRestart = input("Verkeerde input gekregen, wil je het opnieuw proberen? (Y/N) ")
        f.br()

        if answerRestart == 'Y':
            # TrafficDetection.__main__()
            __main__()
            return

        elif answerRestart == 'N':
            print("Afsluiten...")
            sys.exit()

        else:
            print("Verkeerd antwoord, afsluiten...")
            sys.exit()


# De main functie ook daadwerkelijk opstarten...
__main__()