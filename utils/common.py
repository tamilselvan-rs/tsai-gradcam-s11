import torch
from torchvision import datasets, transforms

from torchsummary import summary

def is_cuda_available():
   return torch.cuda.is_available()

def get_dst_device():
    return torch.device("cuda" if is_cuda_available() else "cpu")

def get_train_transforms():
    return transforms.Compose([
        # transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
        # transforms.Resize((28, 28)),
        # transforms.RandomRotation((-15., 15.), fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

def get_test_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])