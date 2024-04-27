import json
import os

import cv2
import torch
from PIL import Image
from matplotlib import cm
from pandas import np
from torch import nn
from torchvision import models
from torchvision.transforms import transforms


def get_data_transformation():
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return data_transform


def json_loader(json_file_path):
    with open(json_file_path, 'r') as file:
        data_store = json.load(file)
        return data_store


def open_video(video_path):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print("error opening video stream or file")
    return capture


def get_color():
    color_labels = {
        'sitting': (0, 255, 255),
        'standing': (255, 255, 0),
        '-1': (0, 0, 255)
    }
    return color_labels


def get_bonding_box_by_frame_id(json_object_, frame_id_):
    # check if frame id match
    if json_object_['Frame'] != frame_id_:
        print('error with frame id')

    number_of_chicken = json_object_['NChickens']
    result = []
    current_ground_truth = json_object_['IDs']
    label = None
    for (k, v) in current_ground_truth.items():
        chicken_id = k
        if 'class' in v:
            cls = v.get('class')

        if 'bbox' in v:
            bbox = v.get('bbox')

        if 'pred' in v:
            label = v.get('pred')

        if 'probability' in v:
            prob = v.get('probability')
        else:
            prob = '0'
        if cls == 'chicken' and label is not None:
            result.insert(len(result),
                          [frame_id_,
                           chicken_id,
                           bbox,
                           label,
                           prob
                           ])
        else:
            result.insert(len(result),
                          [frame_id_,
                           chicken_id,
                           bbox
                           ])

    return result


def draw_frame_text(img_, bbox, color_labels, pred_label, prob):
    color = color_labels[pred_label]
    txt = pred_label+': '+str(round(prob, 2))
    cv2.rectangle(img_,
                  (bbox[0], bbox[3]),
                  (bbox[2], bbox[1]),
                  color,
                  2)
    cv2.putText(img_,
                txt,
                (bbox[0], bbox[1] - 5),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                color,
                1)


def initialize_file_path(video_name,
                         gt_path="../../data/gt_files/",
                         dt_path="../../data/det_files/",
                         data_path="../../data/videos/",
                         out_path="../../data/chicken_state/"):

    video_path = os.path.join(data_path, video_name)
    gt_file = gt_path+'gt-' + video_name[:-4] + '.json'
    dt_file = dt_path+'det-' + video_name[:-4] + '.json'
    file_path_ = {
        "gt_file": gt_file,
        "dt_file": dt_file,
        "gt_path": gt_path,
        "dt_path": dt_path,
        "data_path": data_path,
        "out_path": out_path,
        "video_name": video_name,
        "video_path": video_path,
    }
    return file_path_


def read_class_name(path_name):
    # read csv file as a list of lists
    with open(path_name) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    list = [x.strip() for x in content]
    return list


def predict(img, model, device):
    # print(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_ = Image.fromarray(img)
    data_transform = get_data_transformation()
    input_tensor = data_transform(img_)
    inputs = input_tensor.to(device)
    input_batch = inputs.unsqueeze(0).cuda()
    with torch.no_grad():
        output = model(input_batch)
        p = torch.nn.functional.softmax(output, dim=1)
    value, pred = torch.max(p, 1)
    return value, pred


def draw_frame_from_list(img_, res_list, color_labels):
    for res in res_list:
        draw_frame_text(img_, res[2], color_labels, res[3], res[4])


if __name__ == '__main__':
    # initialise the parameters
    file_path = initialize_file_path("FF_NVR2_Hik_16-09-2020_08-14_to_08-47_25FPS.mp4")
    # load json file
    json_object = json_loader(file_path["dt_file"])
    # open video
    cap = open_video(file_path['video_path'])
    print(file_path['video_path'])
    # num of frame proceed
    num_frame_processed = json_object[0]['metadata']['Frames processed']
    # read class name
    labels = read_class_name('resNet/classes.txt')
    # get state model
    # get colors
    color_labels = get_color()

    # confidence score
    confidence_score = 0.4

    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    weight = torch.load('resNet/model_best.pth')
    model_ft.load_state_dict(weight)
    model_ft.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    # generate json file
    cap = open_video(file_path['video_path'])
    frame_id = 1
    
    while cap.isOpened():
        # open video frame and check whether it opened correctly
        ret, frame = cap.read()
        if not ret:
            break
        # raise Exception("error opening the video frame")
        # set display color
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # img = frame
        # retrieve bounding box by frame id
        json_by_frame = json_object[frame_id]
        json_by_frame_boxes = json_by_frame['IDs']
        result_list = get_bonding_box_by_frame_id(json_by_frame, frame_id)
        # get prediction
        for res in result_list:
            chicken_id = res[1]
            bbox = res[2]
            json_by_bbox = json_by_frame_boxes.get(chicken_id)
            chicken_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            [value, pred] = predict(chicken_img, model_ft, device)
            if value < confidence_score:
                label = '-1'
            else:
                label = labels[pred]
            # update json
            json_by_bbox.update({"pred": label})
            json_by_bbox.update({"probability": value.item()})

        frame_id += 1
        print(frame_id)
        if cv2.waitKey(24) & 0xff == ord('q'):
            break

    # save json
    det_prob_json = file_path['dt_path']+file_path['video_name'][:-4]+'_prob.json'
    print(det_prob_json)
    with open(det_prob_json, 'w', encoding='utf-8') as f:
        json.dump(json_object, f, ensure_ascii=False, indent=4)
    cap.release()
    cv2.destroyAllWindows()

    # write videos
    cap = open_video(file_path['video_path'])
    json_object_n = json_loader(det_prob_json)
    # print(file_path['json_file_det_path'])
    frame_id = 1
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_video = file_path['data_path']+file_path['dt_path'][:-4]+'_prob.mp4'
    out = cv2.VideoWriter('../../data/det_files/det-2_frolicking_prob.mp4', fourcc, 10.0,
                          (frame_width, frame_height))
    while cap.isOpened():
        # open video frame and check whether it opened correctly
        ret, frame = cap.read()
        if not ret:
            break
        # set display color
        # img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img = frame
        # retrieve bounding box by frame id
        result_list = get_bonding_box_by_frame_id(json_object_n[frame_id], frame_id)
        # draw bounding box to the frame
        draw_frame_from_list(img, result_list, color_labels)
        out.write(img)
        cv2.imshow('frame', img)
        frame_id += 1
        if cv2.waitKey(24) & 0xff == ord('q'):
            break
    cap.release()
    out.release()
    # cv2.destroyAllWindows()
