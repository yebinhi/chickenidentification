import cv2
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def get_data_transformation():
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return data_transform


def get_data_loader(validation_path):
    val_image_dataset = ImageFolderWithPaths(validation_path, get_data_transformation())
    data_loader = torch.utils.data.DataLoader(val_image_dataset,
                                              batch_size=16,
                                              shuffle=True,
                                              num_workers=4)
    dataset_sizes = len(val_image_dataset)
    class_names = val_image_dataset.classes
    return data_loader, dataset_sizes, class_names


def save_img_to_folder(original_img, path, name):
    img = cv2.imread(original_img, 1)
    img_path = path + name
    print(img_path)
    cv2.imwrite(img_path, img)


def validate(model):
    [data_loader, dataset_size, class_name] = get_data_loader('../../data/chicken_state/test')
    model.eval()

    running_corrects = 0
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0

    preds_all=[]
    with torch.no_grad():
        for i, (inputs, labels, sample_fnames) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            p = torch.nn.functional.softmax(outputs, dim=1)
            print(torch.nn.functional.softmax(outputs[0], dim=0))
            _, preds = torch.max(outputs, 1)
            # save error image to a folder
            # save_error_image(inputs, labels, preds, sample_fnames, class_name)

            running_corrects += torch.sum(preds == labels.data)
            # tp += torch.sum(preds * labels.data)
            tp += torch.sum((preds * labels.data) != 0)
            # tn += torch.sum((preds - 1) * (labels.data - 1))
            tn += torch.sum(((preds - 1) * (labels.data - 1)) != 0)
            # fp += torch.sum(preds * (labels.data - 1))
            fp += torch.sum((preds * (labels.data - 1)) != 0)
            # fn += torch.sum((preds - 1) * labels.data)
            fn += torch.sum(((preds - 1) * labels.data) != 0)

        acc = running_corrects.double() / dataset_size
        # Precision
        precision = tp / (tp + fp)
        # Recall
        recall = tp / (tp + fn)
        # F1
        f1 = 2 * precision * recall / (precision + recall)
        print('Acc: {:.6f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(
            acc, precision, recall, f1))


def save_error_image(inputs, labels, preds, sample_fnames, class_name):
    for j in range(inputs.size()[0]):
        if preds[j] != labels[j]:
            # current_time = str(get_current_time_stamp())
            original_file_name = sample_fnames[j].rsplit('/', 1)[1]
            name = class_name[preds[j]] + '-' + original_file_name
            save_img_to_folder(sample_fnames[j], '../../data/chicken_state/errors/', name)


if __name__ == '__main__':
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    weight = torch.load('resNet/model_best.pth')
    model_ft.load_state_dict(weight)
    model_ft.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    validate(model_ft)
