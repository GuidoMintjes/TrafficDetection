# Import de coco tools voor de coco dataset ==> (cocodataset.org)
from pycocotools.coco import COCO

import sys

# Allemaal dingen voor plaatjes
from cv2 import cv2
import PIL
import requests
import io

import matplotlib
import matplotlib.pyplot as plt

import pylab

# Dit is ABSOLUUT nodig, anders kan matplotlib geen vensters openen!!!
matplotlib.rcParams['interactive'] = True
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

import numpy as np
import os
import skimage.io as skio


def loadNShowIm(img, catIds):
    coco_url = img['coco_url']
    image_bytes = io.BytesIO(requests.get(coco_url).content)
    
    imDecoded = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), 1)
    

    cv2.imshow('Plaatje', imDecoded)
    cv2.waitKey()




# Maak nog een check voor eerste keer!

datadir = '..'
datatype = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(datadir, datatype)


coco = COCO(annFile)


categories = coco.loadCats(coco.getCatIds())
nms = [category['name'] for category in categories]
nms = set([category['supercategory'] for category in categories])


# Bij de categorie traffic light foto's vinden
print("Alle foto's vinden met stoplichten...")
catIds = coco.getCatIds(catNms=['traffic light'])
imgIds = coco.getImgIds(catIds = catIds)



# Een simpele loop
for i in range(0, 12):

    print("Laad een willekeurig plaatje...")
    img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

    print("Laat het willekeurige plaatje zien...")
    loadNShowIm(img, catIds)
