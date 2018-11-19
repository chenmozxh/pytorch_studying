# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import os
import copy
import torch.nn.functional as F
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Data_loader(Data_Path):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            #transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = Data_Path
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
    #dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return dataloaders, image_datasets, class_names

#dataloaders, _ = Data_loader("hymenoptera_data")
#print(dataloaders['train'][0].cpu())
#print(dataloaders['val'].data())

######################################################################
"""
def imshow(inp, title=None):
    # Imshow for Tensor
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
"""

def train_model(dataloaders, image_datasets, model, criterion, optimizer, scheduler, num_epochs=25):
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                print('{} {} {} {}'.format(type(inputs), inputs.shape, type(labels), labels))
                #print('{} {}'.format(inputs.shape, labels.shape))
                #print('{}'.format(labels))
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
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def train(dataloaders, image_datasets):
    model_ft = models.resnet18(pretrained=True)
    #model_ft = models.resnet34(pretrained=True)
    #model_ft = models.resnet50(pretrained=True)
    #model_ft = models.resnet101(pretrained=True)
    #model_ft = models.resnet152(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(dataloaders, image_datasets, model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

    torch.save(model_ft,"models/best_resnet.pkl")

def test(model_path, dataloaders, image_datasets, class_names):
    #model = models.resnet18(pretrained=True)
    #model.load_state_dict(torch.load(model_path))
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    model = torch.load(model_path)
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            if i > 0: break
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                print('predicted: {}'.format(class_names[preds[j]]))

        print("=================================")

        running_loss, running_corrects = 0.0, 0.0
        criterion = nn.CrossEntropyLoss()
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            if i > 0: break
            inputs = inputs.to(device)
            labels = labels.to(device)

            print(labels.data)
            with torch.set_grad_enabled("val" == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                #prob_ = nn.Softmax(outputs)
                prob_ = F.softmax(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes["val"]
        epoch_acc = running_corrects.double() / dataset_sizes["val"]
        print('{} Loss: {:.4f} Acc: {:.4f}'.format("val", epoch_loss, epoch_acc))

        model.train(mode=was_training)

def test_single_image(model_path, img_path):
    model = torch.load(model_path)
    model.eval()

    with torch.no_grad():
        img = Image.open(img_path).convert("RGB")
        T = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        #img = torch.stack([T(img)], 0)
        img = T(img).unsqueeze(0)
        img = img.to(device)
        outputs = model(img)
        print(outputs)
        prob_ = F.softmax(outputs, 1).cpu().numpy().tolist()
        print("prob_")
        print(prob_)

if __name__ == "__main__":
    dataloaders, image_datasets, class_names = Data_loader('hymenoptera_data')
    train(dataloaders, image_datasets)
    #test("models/best_resnet.pkl", dataloaders, image_datasets, class_names)
    #test_single_image("models/best_resnet.pkl", "hymenoptera_data/val/ants/239161491_86ac23b0a3.jpg")
    #test_single_image("models/best_resnet.pkl", "hymenoptera_data/val/bees/2670536155_c170f49cd0.jpg")


