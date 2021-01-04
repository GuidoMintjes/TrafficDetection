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
import COCO

# import TrafficDetection

# Start met printen naar de console en dingen vragen en dergelijke
f.br()


def __main__():
    print("De COCO 2017 dataset kan automatisch worden gedownload, de bosch dataset zal vooraf getraind worden en de weights te verkrijgen zijn")
    f.br()

    answer = input('Wil je de AI trainen op de bosch dataset (met yolov3), de AI testen op COCO, of de AI testen op andere dingen? (C/Y/A) ')
    f.br()

    if answer == 'C':
        # Open het COCO script
        COCO.__main__()


    elif answer == 'Y':
        # Open het YOLO script
        YOLOv3.__main__()

    elif answer == 'A':

        print("Deze functie is momenteel in ontwikkeling...")
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