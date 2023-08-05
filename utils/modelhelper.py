from tqdm import tqdm
from torchsummary import summary
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np

def print_model_summary(model):
    summary(model, input_size=(3, 32, 32)) 

def get_correct_predict_count(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def get_incorrect_predictions(pPrediction, pLabels):
    incorrect_preds = pPrediction.argmax(dim=1).eq(pLabels)
    incorrect_items = []
    for i in range(len(incorrect_preds)):
        if (incorrect_preds[i] ==  False):
            incorrect_items.append(i)
    return incorrect_items

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def get_missclassification(model, data_loader, device):
    model.eval()
    incorrect_preds = []
    for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            incorrect_preds.append(get_incorrect_predictions(output, target))

    

    out = torch.index_select(data, 0, torch.tensor(incorrect_preds).to(device)).cpu()
    pred_labels = torch.index_select(output.argmax(dim=1), 0, torch.tensor(incorrect_preds).to(device)).cpu()
    correct_labels = torch.index_select(target, 0, torch.tensor(incorrect_preds).to(device)).cpu()
    return [
        out,
        pred_labels,
        correct_labels
    ]

import torchvision
def print_incorrect_preds(model, data_loader, device, inv_normalization: torchvision.transforms.Normalize, count):
    cifar_dict = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
    }

    [out, pred_labels, correct_labels] = get_missclassification(model, data_loader, device)

    fig, axs = plt.subplots(2, 5)
    for j in range(2):
        for i in range(5):
            index_1d = 5*j + i;
            # print(out[index_1d])
            img = out[index_1d]
            img = inv_normalization(image=img)["image"]
            # print(img)
            npimg = img.numpy()
            npimg = np.transpose(npimg, (1,2,0))
            # print(f'{np.amax(npimg)}, {np.amin(npimg)}')
            axs[j, i].imshow(npimg)
            axs[j, i].set_title(f'Predicted: {cifar_dict[int(pred_labels[index_1d].numpy())]}/ Actual: {cifar_dict[int(correct_labels[index_1d].numpy())]}')
    plt.show()

def train_model(model, device, train_loader, optimizer, criterion):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0
  print_errors = True

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss += loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()

    correct += get_correct_predict_count(pred,
                                         target)

    # incorrect_preds = get_incorrect_predictions(pred, target)
    # if (print_errors):
    #     print_incorrect_preds(incorrect_preds, data, pred, target, device)
    #     print_errors = False

    processed += len(data)

    pbar.set_description(
        desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_accuracy = 100 * correct / processed

    train_loss /= len(train_loader)

  return [ train_accuracy, train_loss ]


        

def test_model(model, device, test_loader, criterion):
    model.eval()
    incorrect_preds = []
    incorrect_pred_count = 10
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

            correct += get_correct_predict_count(output, target)
            incorrect_preds = get_incorrect_predictions(output, target)
            #print_incorrect_preds(incorrect_preds, data, output, target)
            
            

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return [test_accuracy, test_loss]

def get_normalization_module(type='bn', num_filters=0, group_size=0):
    if type == 'bn':
        return nn.BatchNorm2d(num_features=num_filters)
    
    if type == 'gn':
        return nn.GroupNorm(num_groups=group_size, num_channels=num_filters)

    if type == 'ln':
        return nn.GroupNorm(num_groups=1, num_channels=num_filters)
    return None

def get_activation_module(type='relu'):
    if type == 'relu':
        return nn.ReLU()

def block(in_channels, num_filters, filter_size=3, padding=0, norm='skip', activation='skip', group_size=0, drop_out=0):

    sequence = nn.Sequential()

    '''
    Add default Convolution step
    #no of filters => output
    filter size => kernel
    '''
    sequence.append(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            padding=padding
        )
    )

    '''
    Add normalization
    Allowed Values:
        `bn` = Batch Normalization
        `gn` = Group Normalization
        `ln` = Layer Normalization
        `skip` = Skip Normalization Step
    '''
    if norm != 'skip':
        sequence.append(
            get_normalization_module(norm, num_filters, group_size=group_size)
        )


    '''
    Add Activation
    Allowed Values:
        `relu` = ReLU Activation Function
        `skip` = Skip Activation Step
    '''
    if activation != 'skip':
        sequence.append(
           get_activation_module(type=activation)
        )

    '''
    Add Dropout
    '''
    if drop_out > 0:
        sequence.append(
            nn.Dropout2d(drop_out)
        )

    return sequence