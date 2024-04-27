"""
created on tue aug 25 09:43:02 2020

@author: bin ye
"""

import json
import os
import cv2
from pathlib import Path

from pandas import np
from csv import reader
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms


def json_loader(json_file_path):
    with open(json_file_path, 'r') as file:
        data_store = json.load(file)
        return data_store


def initialize_file_path(video_name,
                         gt_path="../../data/gt_files/",
                         data_path="../../data/videos/",
                         out_path="../../data/chicken_state/state/"):
    video_path = os.path.join(data_path, video_name)
    json_file_name = 'gt-' + video_name[:-4] + '.json'
    json_file_name_pred = '../../data/chicken_state/json_with_preds.json'
    json_file_path = os.path.join(gt_path, json_file_name)
    file_path_ = {
        "gt_path": gt_path,
        "data_path": data_path,
        "out_path": out_path,
        "video_name": video_name,
        "json_file_name": json_file_name,
        "json_file_name_pred": json_file_name_pred,
        "video_path": video_path,
        "json_file_path": json_file_path
    }
    return file_path_


def open_video(video_path):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print("error opening video stream or file")
    return capture


def get_color():
    color_labels = {
        'sitting': (0, 128, 0),
        'standing': (255, 255, 0),
        'mis': (0, 0, 255)
    }
    return color_labels


def get_bonding_box_by_frame_id(json_object_, frame_id_):
    # check if frame id match
    if json_object_['Frame'] != frame_id_:
        print('error with frame id')

    number_of_chicken = json_object_['NChickens']
    result = []
    current_ground_truth = json_object_['IDs']
    for (k, v) in current_ground_truth.items():
        chicken_id = k
        if 'class' in v:
            cls = v.get('class')
        if 'bbox' in v:
            bbox = v.get('bbox')
        if 'state' in v:
            state = v.get('state')
        if 'fueling' in v:
            fueling = v.get('fueling')
        if 'pred' in v:
            pred_la = v.get('pred')

        if cls == 'chicken' and state and state != '-1':
            result.insert(len(result),
                          [chicken_id,
                           state,
                           bbox,
                           number_of_chicken,
                           frame_id_,
                           pred_la])
    return result


def draw_frame_text(img_, bbox, state_, color_labels, labels, pred_label):

    color = color_labels[state_]
    text =''
    if state_ != pred_label:
        color = color_labels['mis']
        text = state_+' : '+pred_label
        print(text)
    else:
        color = color_labels[state_]
        text = state_

    cv2.rectangle(img_,
                  (bbox[0], bbox[3]),
                  (bbox[2], bbox[1]),
                  color,
                  2)
    cv2.putText(img_,
                state_.upper(),
                (bbox[0], bbox[1] - 5),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                color,
                1)


def draw_frame_from_list(img_, res_list, color_labels, labels):
    for res in res_list:
        draw_frame_text(img_, res[2], res[1], color_labels, labels, res[5])


def read_class_name(path_name):
    # read csv file as a list of lists
    with open(path_name) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    list = [x.strip() for x in content]
    return list


if __name__ == '__main__':
    # initialise the parameters
    file_path = initialize_file_path("Week_2_proof_of_concept.mp4")
    # load json file
    json_object = json_loader(file_path['json_file_name_pred'])
    # open video
    cap = open_video(file_path['video_path'])
    # num of frame proceed
    num_frame_processed = json_object[0]['metadata']['Frames processed']
    # read class name
    labels = read_class_name('resNet/classes.txt')
    # get state model
    # get colors
    color_labels = get_color()

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # read frames
    frame_id = 1
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('../../data/chicken_state/outpy.avi', fourcc, 10.0, (frame_width, frame_height))

    while cap.isOpened():
        # frame process limit
        if frame_id > num_frame_processed:
            break
        # open video frame and check whether it opened correctly
        ret, frame = cap.read()
        if not ret:
            raise Exception("error opening the video frame")
        # set display color
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # retrieve bounding box by frame id
        result_list = get_bonding_box_by_frame_id(json_object[frame_id], frame_id)
        # draw bounding box to the frame
        draw_frame_from_list(img, result_list, color_labels, labels)
        out.write(img)
        cv2.imshow('frame', img)
        frame_id += 1
        if cv2.waitKey(24) & 0xff == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
