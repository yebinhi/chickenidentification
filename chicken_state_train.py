import os
import time

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchsampler import ImbalancedDatasetSampler
from torchvision import models, transforms, datasets


def get_data_transformation():
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


def get_dataloaders(data_dir):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              get_data_transformation()[x])
                      for x in ['train', 'val']}
    data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                   batch_size=16,
                                                   sampler=ImbalancedDatasetSampler(image_datasets[x]),
                                                   num_workers=4)
                    for x in ['train', 'val']}
    return [image_datasets, data_loaders]


def train_model(dataloders, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    use_gpu = torch.cuda.is_available()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    dataset_sizes = {'train': len(dataloders['train'].dataset),
                     'val': len(dataloders['val'].dataset)}

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloders[phase]:
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # running_loss += loss.data[0]
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                train_epoch_loss = running_loss / dataset_sizes[phase]
                train_epoch_acc = running_corrects / dataset_sizes[phase]
            else:
                valid_epoch_loss = running_loss / dataset_sizes[phase]
                valid_epoch_acc = running_corrects / dataset_sizes[phase]

            if phase == 'val' and valid_epoch_acc > best_acc:
                best_acc = valid_epoch_acc
                best_model_wts = model.state_dict()

        print('Epoch [{}/{}] train loss: {:.4f} acc: {:.4f} '
              'valid loss: {:.4f} acc: {:.4f}'.format(
            epoch, num_epochs - 1,
            train_epoch_loss, train_epoch_acc,
            valid_epoch_loss, valid_epoch_acc))

    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    [image_datasets, data_loaders] = get_dataloaders('../../data/chicken_state')
    optimizer = torch.optim.SGD(model_ft.fc.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    start_time = time.time()
    model = train_model(data_loaders, model_ft, criterion, optimizer, exp_lr_scheduler, num_epochs=2)
    print('Training time: {:10f} minutes'.format((time.time() - start_time) / 60))
