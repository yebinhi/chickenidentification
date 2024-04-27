# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 17:12:21 2020

@author: KtDiaz
"""
import sys
import os.path as osp
import json
import os
import time

import logging
import platform
import numpy as np
import cv2

import greengrasssdk

# These directories will be used to keep
# the test image and the libraries used (like numpy and cv2).
# It should be configured as a local resource in GG.
IMG_DIR = "/img"
LIB_DIR = '/lib'
SAVE_DIR = '/save'

# Append the dirs to PATH
sys.path.append(IMG_DIR)
sys.path.append(LIB_DIR)
sys.path.append(SAVE_DIR)

gg_client = greengrasssdk.client('iot-data')

# Initialize logger
logger = logging.getLogger(__name__)

# Retrieving platform information to send from Greengrass Core
my_platform = platform.platform()

# Loading the classifier for frontal face
haar_cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def read_source():
    '''
    Reads the input
    '''
    if osp.isdir(IMG_DIR):
        sourcelist = [osp.join(osp.realpath('.'), IMG_DIR, img) for img in os.listdir(IMG_DIR) if
                      osp.splitext(img)[1] == '.png' or osp.splitext(img)[1] == '.jpeg' or osp.splitext(img)[
                          1] == '.jpg']

    return sourcelist


def draw(img, bb):
    '''
    Draws the detections
    '''
    for (x, y, w, h) in bb:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)

    T = 'Nfaces: ' + str(len(bb))
    cv2.putText(img, T, (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, [0, 0, 255], 1)

    return img


def get_inference(image_filename, scaleFactor=1.2, minNeighbors=5):
    logging.info('Invoking Greengrass Inference Service')

    try:
        # Loading the image to be tested
        test_image = cv2.imread(image_filename)

        # Converting to grayscale
        test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

        # Applying the haar classifier to detect faces
        faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor=scaleFactor, minNeighbors=5)
        Nfaces = str(len(faces_rects))

        test_image = draw(test_image, faces_rects)

        name = osp.basename(image_filename.rstrip('//'))
        cv2.imwrite(osp.join(SAVE_DIR, name), test_image)

        return Nfaces

    except Exception as e:
        logging.error("Failed to publish message: " + repr(e))


# main function
def function_handler(event, context):
    sourcelist = read_source()
    f = 0
    start = time.time()
    for image_filename in sourcelist:
        f += 1
        num_faces = get_inference(image_filename)

        # Creates and sends messages to a topic
        try:
            if not my_platform:
                gg_client.publish(
                    topic="hello/face/detector",
                    queueFullPolicy="AllOrException",
                    payload=json.dumps(
                        {"message": "Hello Face detector! Sent from Greengrass Core."
                                    + ", Frame: {}".format(f)
                                    + " Number of faces in image: {}".format(num_faces)
                                    + ", FPS is {:5.2f}".format(f / (time.time() - start))
                         }
                    ),
                )
            else:
                gg_client.publish(
                    topic="hello/face/detector",
                    queueFullPolicy="AllOrException",
                    payload=json.dumps(
                        {"message": "Hello Face detector! Sent from Greengrass Core running on platform: {}.".format(
                            my_platform)
                                    + ", Frame: {}".format(f)
                                    + " Number of faces in image: {}".format(num_faces)
                                    + ", FPS is {:5.2f}".format(f / (time.time() - start))
                         }
                    ),
                )

        except Exception as e:
            logger.error("Failed to publish message: " + repr(e))

    return
