from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  models, transforms
import copy
#from train_model import train_model
import time
import os
from torch.utils.data import Dataset

from PIL import Image
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# use PIL Image to read image
def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))

# define your Dataset. Assume each line in your .txt file is [name/one space/label], for example:0001.jpg 1
class customData(Dataset):
    def __init__(self, img_path, txt_path, dataset = '', data_transforms=None, loader = default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.strip().split('\t')[0]) for line in lines]
            self.img_label = [int(line.strip().split('\t')[-1]) for line in lines]
            #self.img_name = [os.path.join(img_path, line.strip()[:-2]) for line in lines]
            #self.img_label = [int(line.strip()[-1:]) for line in lines]
        #print(len(self.img_name))
        #print(self.img_name)
        #print(len(self.img_label))
        #print(self.img_label)
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label

#    train_model(dataloaders, image_datasets, model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
def train_model(dataloaders, image_datasets, model, criterion, optimizer, scheduler, num_epochs=25):
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 30)
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
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to(device)
    #model_ft = torch.nn.DataParallel(model_ft)#, device_ids=[0,1])
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(dataloaders, image_datasets, model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
    torch.save(model_ft,"models/best_resnet.pkl")


def Data_loader():
    batch_size = 4
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
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: customData(img_path='hymenoptera_data_cp/',
                                    txt_path=(x + '.txt'),
                                    data_transforms=data_transforms,
                                    dataset=x) for x in ['train', 'val']}

    # wrap your data and label into Tensor
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=batch_size,
                                                 shuffle=True) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return image_datasets, dataloaders

def Test_dataloaders(dataloaders):
    print("train"*20)
    for inputs, labels in dataloaders['train']:
        print('{} {} {} {}'.format(type(inputs), inputs.shape, type(labels), labels.detach()))
        #labels = torch.IntTensor([labels])
        #print('{} {} {}'.format(type(inputs), inputs.shape, type(labels), labels.detach()))
        #print('{}'.format(labels))
    print("test"*20)
    for inputs, labels in dataloaders['val']:
        print('{} {} {} {}'.format(type(inputs), inputs.shape, type(labels), labels.detach()))

if __name__ == '__main__':
    image_datasets, dataloaders = Data_loader()
    #Test_dataloaders(dataloaders)
    train(dataloaders, image_datasets)


