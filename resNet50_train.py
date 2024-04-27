from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torchsampler import ImbalancedDatasetSampler
from sklearn.metrics import classification_report


def get_data_transformation():
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(25),
            transforms.ColorJitter(0.4, 0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms


def initialize_parameters(data_dir='../../data/chicken_state'):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              get_data_transformation()[x])
                      for x in ['train', 'val']}
    data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                   batch_size=16,
                                                   # sampler=ImbalancedDatasetSampler(image_datasets[x]),
                                                   num_workers=4)
                    for x in ['train', 'val']}
    # torch.save(data_loaders['val'], 'data_loaders.pt')
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    class_names = image_datasets['train'].classes
    return [data_loaders, dataset_sizes, class_names]


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.01)  # pause a bit so that plots are updated


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def train_model(model, criterion, data_loaders, dataset_sizes, optimizer, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # [data_loaders, dataset_sizes, class_names] = initialize_parameters()

    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    # total_step = len(data_loaders['train'])

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            true_labels = []
            pred_labels = []
            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                pred_labels.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                # scheduler.step()
                train_acc.append(epoch_acc)
                train_loss.append(epoch_loss)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                print(best_acc.cpu())
                best_model_wts = copy.deepcopy(model.state_dict())

            print('{} Loss: {:.4f} Acc: {:.6f}'.format(phase, epoch_loss, epoch_acc))
            print(classification_report(true_labels, pred_labels))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, 'resNet/model_best.pth')
    return model


def visualize_model(model, num_images=6):
    [data_loaders, dataset_sizes, class_names] = initialize_parameters()
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])
                time.sleep(1)
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def num_of_layers_to_freeze(model, num):
    #  freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    #  unfreeze some layers
    temp = 0
    for param in model.parameters():
        temp += 1
        if temp >= num:
            param.requires_grad = True
    return model


def save_result(df_obj, path, file_name):
    df_obj.to_csv(path + '40layers.csv', index=False)


if __name__ == '__main__':
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    # nn.BCELoss()
    [data_loaders, dataset_sizes, class_names] = initialize_parameters()
    #
    inputs, classes = next(iter(data_loaders['train']))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    # imshow(out, title=[class_names[x] for x in classes])

    # time.sleep(10000)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # optimizer = torch.optim.Adam(model_ft.parameters(), lr=0.001)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = num_of_layers_to_freeze(model_ft, 120)
    model_ft = train_model(model_ft, criterion, data_loaders, dataset_sizes, optimizer_ft,
                           num_epochs=50)
