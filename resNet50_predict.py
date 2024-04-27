import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


def read_class_name(path_name):
    # read csv file as a list of lists
    with open(path_name) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    list = [x.strip() for x in content]
    return list


def get_data_transformation():
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return data_transform


def predict_with_path(img_path, model, device):
    input_image = Image.open(img_path)
    data_transform = get_data_transformation()
    input_tensor = data_transform(input_image)
    inputs = input_tensor.to(device)
    input_batch = inputs.unsqueeze(0).cuda()
    with torch.no_grad():
        output = model(input_batch)
        p = torch.nn.functional.softmax(output, dim=1)
    value, pred = torch.max(p, 1)
    return value, pred


def predict(img, model, device):
    data_transform = get_data_transformation()
    input_tensor = data_transform(img)
    inputs = input_tensor.to(device)
    input_batch = inputs.unsqueeze(0).cuda()
    with torch.no_grad():
        output = model(input_batch)
        p = torch.nn.functional.softmax(output, dim=1)
    value, pred = torch.max(p, 1)
    return value, pred


if __name__ == '__main__':
    # load model structure
    model_ft = models.resnet50(pretrained=True)
    # config structure
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    # load pre-trained weight
    weight = torch.load('resNet/model_best.pth')
    model_ft.load_state_dict(weight)
    # set to evaluation
    model_ft.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    # read class name
    class_names = read_class_name('resNet/classes.txt')
    img_path = '../../data/chicken_state/test/sitting/sitting_476_1.png'
    [value, pred] = predict_with_path(img_path, model_ft, device)
    print('The chicken is: '+class_names[pred] + ', probability: '+str(value.cpu().numpy()[0]))


