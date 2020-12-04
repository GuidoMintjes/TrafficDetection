# Import de coco tools voor de coco dataset ==> (cocodataset.org)
from pycocotools.coco import COCO

import sys
import importlib
from functions import *

import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.io as io

# Eigen python packages
import YOLOv3
import COCO

# Start met printen naar de console en dingen vragen en dergelijke
br()



print("De COCO 2017 dataset kan automatisch worden gedownload, de bosch dataset zal vooraf getraind worden en de weights te verkrijgen zijn")
br()

answer = input('Wil je de AI trainen op de bosch dataset (mey yolov3), de AI testen op COCO, of de AI testen op andere dingen? (C/Y/A) ')
br()

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
    br()

    if answerRestart == 'Y':
        importlib.import_module("TrafficDetection.py")

    elif answerRestart == 'N':
        print("Afsluiten...")
        sys.exit()

    else:
        print("Verkeerd antwoord, afsluiten...")
        sys.exit()