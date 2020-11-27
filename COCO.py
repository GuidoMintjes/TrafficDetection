# Import de coco tools voor de coco dataset ==> (cocodataset.org)
from pycocotools.coco import COCO

import sys

import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.io as io

# Maak nog een check voor eerste keer!

datadir = '..'
datatype = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(datadir, datatype)


coco = COCO(annFile)


categories = coco.loadCats(coco.getCatIds())
nms = [category['name'] for category in categories]
nms = set([category['supercategory'] for category in categories])


catIds = coco.getCatIds(catNms=['traffic light'])
imgIds = coco.getImgIds(catIds = catIds)
# img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]


Image = io.imread(img['coco_url'])
plt.axis('off')
plt.imshow(Image)
plt.plot()