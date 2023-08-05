from torchvision import datasets
import CifarDataSet
import torch

def setup_train_loader(destination, train_transforms):
    batch_size = 128
    kwargs = {
        'batch_size': batch_size, 
        'shuffle': True, 
        'num_workers': 2, 
        'pin_memory': True
    }
    train_data = CifarDataSet.CifarDataSet(destination, train=True, download=True, transform=train_transforms)
    return torch.utils.data.DataLoader(train_data, **kwargs)

def setup_test_loader(destination, test_transforms):
    batch_size = 128
    kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 2,
        'pin_memory': True
    }
    test_data = CifarDataSet.CifarDataSet(destination, train=False, download=True, transform=test_transforms)
    return torch.utils.data.DataLoader(test_data, **kwargs)
