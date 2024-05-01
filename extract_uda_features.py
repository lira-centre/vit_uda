import numpy as np
import torch
import torchvision
import time
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torch.fx import symbolic_trace
import torchvision.transforms as transforms
from M3SDA.datasets.datasets_ import Dataset
import csv
import os
import sys
from typing import Sequence


DATASET = sys.argv[1]#'UDA_CLIPART'
RESOLUTION = 224
MODEL = 'resnet152'#'dinov2_vitg14'#'vit_h14'
SAVE_TRAINING_DATA = True 

def feature_extraction(img, model):
    img = img.to(device)
    if 'dino' in MODEL: 
        features = model (img.to('cuda'))
        print (features.shape)
        return features
    print ('img.shape:', img.shape)
    #print (get_graph_node_names (model))
    if 'resnet' in MODEL:
        feature_extractor = create_feature_extractor(model, return_nodes=['flatten'])
    else:
        feature_extractor = create_feature_extractor(model, return_nodes=['getitem_4'])
    with torch.no_grad():
        out = feature_extractor(img)
    if 'resnet' in MODEL: 
        return out['flatten']
    else:
        return out['getitem_4']

#the function from https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def make_normalize_transform(
            mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
                std: Sequence[float] = IMAGENET_DEFAULT_STD,
                ) -> transforms.Normalize:
        return transforms.Normalize(mean=mean, std=std)

def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_features, test_features = [], []
    train_labels, test_labels = [], []
    weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    if ('dino' in MODEL):
        model = torch.hub.load('facebookresearch/dinov2', model=MODEL)
    elif ('resnet' in MODEL):
        model = torch.hub.load('pytorch/vision:v0.10.0', MODEL, pretrained=True)
    else:
        model = torch.hub.load("facebookresearch/swag", model=MODEL)
    
    model.eval()
    model = model.to(device)
    dir_train = '../UnsupervisedDomainAdaptation'
    dir_test = '../UnsupervisedDomainAdaptation'

    if DATASET == 'UDA_SKETCH':
        labels_file_train = '../UnsupervisedDomainAdaptation/sketch_train.txt'
        labels_file_test = '../UnsupervisedDomainAdaptation/sketch_test.txt'
    elif DATASET == 'UDA_REAL':
        labels_file_train = '../UnsupervisedDomainAdaptation/real_train.txt'
        labels_file_test = '../UnsupervisedDomainAdaptation/real_test.txt'
    elif DATASET == 'UDA_QUICKDRAW':
        labels_file_train = '../UnsupervisedDomainAdaptation/quickdraw_train.txt'
        labels_file_test = '../UnsupervisedDomainAdaptation/quickdraw_test.txt'
    elif DATASET == 'UDA_PAINTING':
        labels_file_train = '../UnsupervisedDomainAdaptation/painting_train.txt'
        labels_file_test = '../UnsupervisedDomainAdaptation/painting_test.txt'
    elif DATASET == 'UDA_INFOGRAPH':
        labels_file_train = '../UnsupervisedDomainAdaptation/infograph_train.txt'
        labels_file_test = '../UnsupervisedDomainAdaptation/infograph_test.txt'
    elif DATASET == 'UDA_CLIPART':
        labels_file_train = '../UnsupervisedDomainAdaptation/clipart_train.txt'
        labels_file_test = '../UnsupervisedDomainAdaptation/clipart_test.txt'
    data_train = []
    labels_train = []
    with open(labels_file_train, 'rt') as f:
        images_labels_train_reader = csv.reader(f, delimiter=' ')
        for row in images_labels_train_reader:
            data_train.append(os.path.join(dir_train, row[0]))
            labels_train.append(row[1])
    data_test = []
    labels_test = []
    with open(labels_file_test, 'rt') as f:
        images_labels_test_reader = csv.reader(f, delimiter=' ')
        for row in images_labels_test_reader:
            data_test.append(os.path.join(dir_test, row[0]))
            labels_test.append(row[1]) 

    trainset = Dataset(data=data_train, label=labels_train, transform=make_classification_eval_transform())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=False, num_workers=4)
    testset = Dataset(data=data_test, label=labels_test, transform=make_classification_eval_transform())
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=4)
 
    start = time.time()
    if SAVE_TRAINING_DATA: 
        i = 0 
        with torch.no_grad():
            for _, inputs_targets in enumerate(trainloader):
                inputs = inputs_targets[0]
                targets = list(inputs_targets[1])
                inputs = inputs.to(device)
                train_features.extend(feature_extraction(inputs,model).detach().cpu().numpy())
                train_labels.extend(targets)
                i = len(train_labels)
                print(i)        
        end = time.time()
    
        with open(f'train_features_{MODEL}_{DATASET}.npy', 'wb') as f:
            np.save(f, np.array(train_features))
        with open(f'train_labels_{MODEL}_{DATASET}.npy', 'wb') as f:
            np.save(f, np.array(train_labels))
        print ("###################### training feature extraction ####################")
        print("Training feature extraction Time: ",round(end - start,2), "seconds")


    start = time.time()

    i = 0
    with torch.no_grad():
        for _, inputs_targets in enumerate(testloader):
            inputs = inputs_targets[0]
            targets = inputs_targets[1]
            inputs = inputs.to(device)
            test_features.extend(feature_extraction(inputs,model).detach().cpu().numpy())
            test_labels.extend(targets)
            i = len(test_labels)
            print(i)
    end = time.time()

    with open(f'test_features_{MODEL}_{DATASET}.npy', 'wb') as f:
        np.save(f, np.array(test_features))
    with open(f'test_labels_{MODEL}_{DATASET}.npy', 'wb') as f:
        np.save(f, np.array(test_labels))
    print ("###################### training feature extraction ####################")
    print("Testing feature extraction Time: ",round(end - start,2), "seconds")
