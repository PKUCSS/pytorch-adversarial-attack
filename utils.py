import torch
import torchvision.transforms as transforms
from torchvision import datasets
from loguru import logger
from models import *


def get_model(model_type):
    if model_type == 'smallCNN':
        return SmallCNN()
    elif model_type == 'preActResnet18':
        return PreActResNet18()
    elif model_type == 'resnet18':
        return ResNet18()
    elif model_type == 'vgg11':
        return  VGG('VGG11')
    elif model_type == 'lenet':
        return LeNet()
    else:
        raise NotImplementedError

def load_model(model, path):
    check_point = torch.load(path)
    try:
        model.load_state_dict(check_point['state_dict'])
    except:
        model.load_state_dict(check_point)
    logger.info("model loaded from {}".format(path))
    return model 

def get_dataloader(dataset, batch_size, data_root='./dataset', train=False):
    transform_test = transforms.Compose([transforms.ToTensor()])
    if dataset == 'mnist':
        data_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=data_root, train=train, download=True, transform=transform_test),
            batch_size=batch_size
        )
    elif dataset == 'cifar10':
        data_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=data_root, train=train, download=True, transform=transform_test),
            batch_size=batch_size
        )
    else:
        raise NotImplementedError
    
    return data_loader

