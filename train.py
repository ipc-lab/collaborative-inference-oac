'''Train CIFAR10 with PyTorch. Took parts of the code from: https://github.com/kuangliu/pytorch-cifar''' 
import os
from turtle import forward
from utils import seed_everything
seed_everything(1)
from torch.cuda.amp import GradScaler
from torch import autocast
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.utils import shuffle
import argparse
from itertools import product
from utils import progress_bar
from argparse import ArgumentParser

weights = torchvision.models.mobilenet.MobileNet_V3_Large_Weights.IMAGENET1K_V2

class Model(nn.Module):
    
    def __init__(self, num_classes):
        super(Model, self).__init__()

        self.num_classes = num_classes
        self.mobilenet = torchvision.models.mobilenet_v3_large(weights=weights)
        self.mobilenet.classifier[3] = nn.Linear(self.mobilenet.classifier[3].in_features, num_classes)
    
    def forward(self, x):
        
        res = self.mobilenet(x)
        
        # res = nn.functional.softmax(res, dim=1)

        return res

"""
class Model(nn.Module):
    
    def __init__(self, num_classes):
        super(Model, self).__init__()

        self.num_classes = num_classes
        self.model = torchvision.models.efficientnet_v2_s(weights=torchvision.models.efficientnet.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)
    
    def forward(self, x):
        
        res = self.model(x)
        
        return res
"""

# Training
def train(epoch, trainloader, net, criterion, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    #scaler = GradScaler()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        #with autocast(device_type='cuda', dtype=torch.float16):
        outputs = net(inputs)
        
        loss = criterion(outputs, targets)

        #loss = scaler.scale(loss)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def eval_on_data(dataloader, net, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    y_pred_beliefs = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            y_pred.append(predicted)
            y_true.append(targets)
            y_pred_beliefs.append(outputs)

            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    res = torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0), torch.cat(y_pred_beliefs, dim=0)
    
    print(res[0].shape, res[1].shape, res[2].shape)
    
    return res

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def gray2rgb(image):
    return image.repeat(3, 1, 1)
"""
rgb_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

gray_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(gray2rgb),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
"""
rgb_transform = {
    "train": transforms.Compose([
        weights.transforms(antialias=True),
        #transforms.RandomHorizontalFlip(),
    ]),
    "eval": weights.transforms(antialias=True)
}

gray_transform = {
    "train": transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(gray2rgb),
        weights.transforms(antialias=True),
        #transforms.RandomHorizontalFlip(),
    ]),
    "eval": transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(gray2rgb),
        weights.transforms(antialias=True)
    ])
}

datasets = {
    "cifar10": {
        "num_classes": 10,
        "cls": torchvision.datasets.CIFAR10,
        "transform": rgb_transform,
    },
    "cifar100": {
        "num_classes": 100,
        "cls": torchvision.datasets.CIFAR100,
        "transform": rgb_transform,
    },
    "mnist": {
        "num_classes": 10,
        "cls": torchvision.datasets.MNIST,
        "transform": gray_transform,
    },
    "fashionmnist": {
        "num_classes": 10,
        "cls": torchvision.datasets.FashionMNIST,
        "transform": gray_transform,
    },
    "food101": {
        "num_classes": 101,
        "cls": torchvision.datasets.Food101,
        "transform": rgb_transform,
    },
    "dtd": {
        "num_classes": 47,
        "cls": torchvision.datasets.DTD,
        "transform": rgb_transform,
    },
    "country211": {
        "num_classes": 211,
        "cls": torchvision.datasets.Country211,
        "transform": rgb_transform,
    },
    "flowers102": {
        "num_classes": 102,
        "cls": torchvision.datasets.Flowers102,
        "transform": rgb_transform,
    },
    "oxford3tpets":{
        "num_classes": 37,
        "cls": torchvision.datasets.OxfordIIITPet,
        "transform": rgb_transform,
    },
    "multiview_oxford3tpets":{
        "num_classes": 37,
        "cls": torchvision.datasets.OxfordIIITPet,
        "transform": rgb_transform,
    },
    
    ## iNaturalist ** TOO MANY IMAGES
    # StanfordCars --NOT WORKING
}

def multiview_process(img, device_idx):
    # x: (B, C, H, W)
    
    # divide x into 20 parts and choose the device_idx-th part

    img_width = img.size[0]
    img_height = img.size[1]
    
    x_sliding =  img_width // 10
    x_size = img_width // 2
    x_idx = device_idx // 4
    
    y_sliding = img_height // 6
    y_size = img_height // 2
    y_idx = device_idx % 4
    
    res = img.crop((x_sliding*x_idx, y_sliding*y_idx, x_sliding*x_idx+x_size, y_sliding*y_idx+y_size))

    return res
    
def train_and_save(data_name, num_devices, num_repeats, num_epochs, root_dir="."):
    seed_everything(1)
    dataset = datasets[data_name]
    
    data_dir = os.path.join(root_dir, 'data')
    results_dir = os.path.join(root_dir, 'results')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    is_multiview = data_name.startswith("multiview_")
    
    train_transform = dataset["transform"]["train"]
    eval_transform = dataset["transform"]["eval"]

    if data_name in ["food101", "stanfordcars", "dtd", "country211", "flowers102"]:
        trainset = dataset["cls"](root=data_dir, split="train", download=True, transform=train_transform)
        testset = dataset["cls"](root=data_dir, split="test", download=True, transform=eval_transform)
    elif data_name in ["oxford3tpets", "multiview_oxford3tpets"]:
        trainset = dataset["cls"](root=data_dir, split="trainval", download=True, transform=train_transform)
        testset = dataset["cls"](root=data_dir, split="test", download=True, transform=eval_transform)
    else:
        trainset = dataset["cls"](root=data_dir, train=True, download=True, transform=train_transform)
        testset = dataset["cls"](root=data_dir, train=False, download=True, transform=eval_transform)

    shuffled_indices = shuffle(np.arange(len(trainset)))
    
    num_traindata = int(len(shuffled_indices)*0.9)
    
    val_inds = shuffled_indices[num_traindata:]
    valset = Subset(trainset, val_inds)
    valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    
    for seed_idx in range(num_repeats):
        seed_everything(seed_idx)

        if is_multiview:
            train_indices = np.expand_dims(shuffled_indices[:num_traindata], 0).repeat(num_devices, axis=0)
        else:
            train_indices = np.array_split(shuffled_indices[:num_traindata], num_devices)
        
        for device_idx, inds in enumerate(train_indices):
            seed_everything(seed_idx)

            print("Device", device_idx)
            trainloader = torch.utils.data.DataLoader(Subset(trainset, inds), batch_size=128, shuffle=True, num_workers=2)
            
            if is_multiview:
                train_transform_mv = transforms.Compose([
                    transforms.Lambda(lambda x: multiview_process(x, device_idx)),
                    train_transform
                ])
                eval_transform_mv = transforms.Compose([
                    transforms.Lambda(lambda x: multiview_process(x, device_idx)),
                    eval_transform
                ])
                        
                trainset = dataset["cls"](root=data_dir, split="trainval", download=True, transform=train_transform_mv)
                testset = dataset["cls"](root=data_dir, split="test", download=True, transform=eval_transform_mv)
                
                valset = Subset(trainset, val_inds)
                
                valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False, num_workers=2)
                testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
                trainloader = torch.utils.data.DataLoader(Subset(trainset, inds), batch_size=128, shuffle=True, num_workers=2)
                
            # Model
            net = Model(num_classes = dataset["num_classes"])
            net = net.to(device)
            if device == 'cuda':
                net = torch.nn.DataParallel(net)
                cudnn.benchmark = True

            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            
            optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

            for epoch in range(num_epochs):
                train(epoch, trainloader, net, criterion, optimizer)
                scheduler.step()
            
            y_train_true, y_train_pred, y_train_pred_beliefs = eval_on_data(trainloader, net, criterion)
            y_val_true, y_val_pred, y_val_pred_beliefs = eval_on_data(valloader, net, criterion)
            y_test_true, y_test_pred, y_test_pred_beliefs = eval_on_data(testloader, net, criterion)

            res = {
                "model": net.state_dict(),
                "inds": inds,
                "device_idx": device_idx,
                "y_train_true": y_train_true,
                "y_train_pred": y_train_pred,
                "y_train_pred_beliefs": y_train_pred_beliefs,
                "y_val_true": y_val_true,
                "y_val_pred": y_val_pred,
                "y_val_pred_beliefs": y_val_pred_beliefs,
                "y_test_true": y_test_true,
                "y_test_pred": y_test_pred,
                "y_test_pred_beliefs": y_test_pred_beliefs
            }

            targetdir = f"results/{data_name}_{num_devices}devices_seed{seed_idx}"
            if not os.path.isdir(targetdir):
                os.makedirs(targetdir)
            
            torch.save(res, f'{targetdir}/{device_idx}.pth')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", choices=["cifar10", "fashionmnist", "mnist", "cifar100", "food101", "stanfordcars", "dtd", "country211", "flowers102", "oxford3tpets", "multiview_oxford3tpets"], required=True, type=str)
    parser.add_argument("--num_repeats", default=5, type=int)
    parser.add_argument("--num_devices", default=20, type=int)
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--root_dir", default=".", type=str)
    
    cfg = vars(parser.parse_args())
    
    train_and_save(cfg["data"], cfg["num_devices"], cfg["num_repeats"], cfg["num_epochs"], cfg["root_dir"])

"""
CUDA_VISIBLE_DEVICES=0 python train.py --data cifar10 --num_repeats 5 --num_devices 20 --num_epochs 50 && CUDA_VISIBLE_DEVICES=0 python train.py --data cifar100 --num_repeats 5 --num_devices 20 --num_epochs 50
CUDA_VISIBLE_DEVICES=1 python train.py --data fashionmnist --num_repeats 5 --num_devices 20 --num_epochs 50 && CUDA_VISIBLE_DEVICES=1 python train.py --data mnist --num_repeats 5 --num_devices 20 --num_epochs 50

CUDA_VISIBLE_DEVICES=1 python train.py --data dtd --num_repeats 5 --num_devices 20 --num_epochs 50 --root_dir=/home/sfy21/oac-based-private-ensembles-local/ &&  CUDA_VISIBLE_DEVICES=1 python train.py --data food101 --num_repeats 5 --num_devices 20 --num_epochs 50 --root_dir=/home/sfy21/oac-based-private-ensembles-local/

---
CUDA_VISIBLE_DEVICES=1 python train.py --data country211 --num_repeats 5 --num_devices 20 --num_epochs 50 --root_dir=/home/sfy21/oac-based-private-ensembles-local/

CUDA_VISIBLE_DEVICES=0 python train.py --data oxford3tpets --num_repeats 5 --num_devices 20 --num_epochs 50 --root_dir=/home/sfy21/oac-based-private-ensembles-local/
CUDA_VISIBLE_DEVICES=1 python train.py --data flowers102 --num_repeats 5 --num_devices 20 --num_epochs 50 --root_dir=/home/sfy21/oac-based-private-ensembles-local/

# python train.py --data multiview_oxford3tpets --num_repeats 5 --num_devices 20 --num_epochs 50
"""