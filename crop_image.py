"""
created on tue aug 25 09:43:02 2020

@author: bin ye
"""

import json
import os
import cv2
from pathlib import Path


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
    json_file_path = os.path.join(gt_path, json_file_name)
    file_path_ = {
        "gt_path": gt_path,
        "data_path": data_path,
        "out_path": out_path,
        "video_name": video_name,
        "json_file_name": json_file_name,
        "video_path": video_path,
        "json_file_path": json_file_path
    }
    return file_path_


def open_video(video_path):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print("error opening video stream or file")
    return capture


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

        if cls == 'chicken' and state and state != '-1':
            result.insert(len(result),
                          [chicken_id,
                           state,
                           bbox,
                           number_of_chicken,
                           frame_id_])
    return result


def draw_frame_text(img_, bbox, state_):
    cv2.rectangle(img,
                  (bbox[0], bbox[3]),
                  (bbox[2], bbox[1]),
                  (0, 255, 255),
                  2)
    cv2.putText(img_,
                state_.upper(),
                (bbox[0], bbox[1]-5),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (0, 0, 255),
                1)


def draw_frame_from_list(img_, res_list):
    for res in res_list:
        draw_frame_text(img_, res[2], res[1])


def crop_images(image, res_list, out_path):
    num = 1
    for res in res_list:
        state = res[1]
        frame_id_ = res[4]
        chicken_id = res[0]
        if state == -1:
            continue
        bbox = res[2]
        image_crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        image_path = out_path + state + '/'
        Path(image_path).mkdir(parents=True, exist_ok=True)
        image_name = state + '_' + str(frame_id_) + '_' + str(chicken_id) + '.png'
        image_path_and_name = image_path + image_name
        if not cv2.imwrite(image_path_and_name, image_crop):
            raise Exception("Could not write image "+image_path_and_name)
        num += 1


if __name__ == '__main__':
    # initialise the parameters
    file_path = initialize_file_path("Week_2_proof_of_concept.mp4")
    # load json file
    json_object = json_loader(os.path.join(file_path['gt_path'], file_path['json_file_name']))
    # open video
    cap = open_video(file_path['video_path'])
    # num of frame proceed
    num_frame_processed = json_object[0]['metadata']['Frames processed']
    # read frames
    frame_id = 1
    while cap.isOpened():
        # frame process limit
        if frame_id > num_frame_processed:
            break
        # open video frame and check whether it opened correctly
        ret, frame = cap.read()
        if not ret:
            raise Exception("error opening the video frame")
        # set display color
        img = frame
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # retrieve bounding box by frame id
        result_list = get_bonding_box_by_frame_id(json_object[frame_id], frame_id)
        # crop the bounding box to a folder
        crop_images(img, result_list, file_path['out_path'])
        # draw bounding box to the frame
        draw_frame_from_list(img, result_list)
        cv2.imshow('frame', img)
        frame_id += 1
        if cv2.waitKey(24) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
