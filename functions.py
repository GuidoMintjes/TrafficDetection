import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import requests, sys, time, datetime
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle

import random

from cv2 import cv2
import numpy as np

colorG = 255

def br():
    print('\n')


def download(url: str, dest_folder: str, chunk_size: int):

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # Maak de map aan als die nog niet bestaat
    
    bestandsNaam = url.split('/')[-1].replace(" ", "_") # -1 om het laatste ding van de splits te krijgen,
    # het replacen zodat we geen spaties krijgen in de bestandsnaam
    bestandsPad = os.path.join(dest_folder, bestandsNaam)

    if os.path.exists(bestandsPad):

        print("Het bestand bestaat al! Er wordt geprobeerd het te overschrijven, mits dit niet lukt, moet u het zelf verwijderen:")
        print(os.path.abspath(bestandsPad))
        br()

    request = requests.get(url, stream = True)

    # Controleer of de request is geaccepteerd
    if request.ok:
        print("Bestand downloaden naar ", os.path.abspath(bestandsPad), " === dit kan even duren...")
        beginTijd = time.time()
        with open(bestandsPad, 'wb') as bestand:
            
            total_length = int(request.headers.get('content-length'))
            totaleChunks = round(total_length / chunk_size)
            
            dl = 0

            chunkNo = 0

            for chunk in request.iter_content(chunk_size = chunk_size):

                if chunk:
                    chunkNo += 1
                    
                    dl += len(chunk)

                    bestand.write(chunk)
                    bestand.flush()
                    os.fsync(bestand.fileno())


                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s] chunk %s van de %s chunks..." % ('=' * done, ' ' * (50 - done), str(chunkNo), str(totaleChunks)))
                    sys.stdout.flush()

        eindTijd = time.time()
        downloadDuur = datetime.timedelta(seconds=(eindTijd - beginTijd))
        print("Download is gelukt! Het duurde {}!".format(str(downloadDuur)))
        br()
    
    else:
        print("Download niet gelukt: status code {}\n{}".format(request.status_code, request.text))
        print("Probeer het bestand zelf te downloaden via {} en in de map {} te zetten!".format(url, dest_folder))


def folderCheck(folder: str):

    if not os.path.exists(folder):
        os.makedirs(folder)  # Maak de map aan als die nog niet bestaat


def load_image_pixels(filename, shape):
    imageOrig = load_img(filename)
    width, height = imageOrig.size


    image = load_img(filename, target_size=shape) # Zebra plaatje als non copyright voorbeeld
    image = img_to_array(image) # Verander het plaatje in een numpy array


    image = image.astype('float32') # Scale de waardes van het plaatje naar tussen de 0 en de 1
    image /= 255.0                  # ^^^

    return image, width, height


def load_image_pixels_video(image, width, height):

    heighte, widthe, colors = image.shape


    shape = (int(width), int(height))


    image = cv2.resize(image, shape)
    image = img_to_array(image) # Verander het plaatje in een numpy array


    image = image.astype('float32') # Scale de waardes van het plaatje naar tussen de 0 en de 1
    image /= 255.0                  # ^^^

    return image, widthe, heighte


# pak alle bounding boxes boven die thresh(hold)
def get_boxes(boxes, labels, thresh):
	v_boxes, v_labels, v_scores = list(), list(), list()
	# ga langs alle boxen heen
	for box in boxes:
		# ga langs alle labels heen (labels worden gemaakt in yolo script, zijn de dingen die je wilt herkennen)
		for i in range(len(labels)):
			
			if box.classes[i] > thresh:     # kijk of die bounding box ook inderdaad over de threshhold heen komt
				v_boxes.append(box)
				v_labels.append(labels[i])
				v_scores.append(box.classes[i]*100)

	return v_boxes, v_labels, v_scores


def randomize(min, max):
    integerR = random.randint(min, max)
    return integerR


def draw_boxes(filename, v_boxes, v_labels, v_scores, obj_thresh):

    image = cv2.imread(filename)

    for i in range(len(v_boxes)):
        box = v_boxes[i]

        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        width, height = x2 - x1, y2 - y1
    
        label = "%s %.2f" % (v_labels[i], (box.get_score() * 100))

        colorR = 255 - (255*box.get_score())
        colorG = 255*box.get_score()


        cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), (colorR, colorG,0), 3)
        cv2.putText(image, 
            label, 
            (box.xmin, box.ymin - 13), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.002 * image.shape[0], 
            (0,colorG,colorB), 2)

    return image


def draw_boxes_video(image, v_boxes, v_labels, v_scores, obj_thresh):

    
    for i in range(len(v_boxes)):
        box = v_boxes[i]

        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        width, height = x2 - x1, y2 - y1
    
        label = "%s %.2f" % (v_labels[i], (box.get_score() * 100))

        colorR = 255 - (255*box.get_score())
        colorG = 255*box.get_score()


        cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), (colorR, colorG, 0), 3)
        cv2.putText(image, 
            label, 
            (box.xmin, box.ymin - 13), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.002 * image.shape[0], 
            (0,colorG,colorB), 2)


    return image