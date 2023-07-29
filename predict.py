
#imports
import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict
import torch.nn.functional as F
from PIL import Image
import glob,os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('path_to_image',type=str)
parser.add_argument('checkpoint', type=str)
parser.add_argument('--top_k', type=int, required=False, default=1)
parser.add_argument('--category_names', type=str, required=False)
parser.add_argument('--gpu',action='store_true')

args = parser.parse_args()


# define load_model

def load_model(filepath):
    checkpoint = torch.load(filepath)
    
    model = models.vgg11(pretrained=True)
    model.classifier = nn.Sequential(nn.Linear(25088, checkpoint['hidden_units']),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(checkpoint['hidden_units'], 102),
                                      nn.LogSoftmax(dim=1))
    
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded.')
    model.class_to_idx = checkpoint['class_to_idx']
    return model


model = load_model(args.checkpoint)


# preprocess image


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    '''
    # Resize the image while keeping the aspect ratio
    image = Image.open(image)
    shortest_side = min(image.size)
    new_size = (256, int(256 * image.size[1] / image.size[0])) if image.size[0] < image.size[1] else (int(256 * image.size[0] / image.size[1]), 256)
    image = image.resize(new_size, Image.ANTIALIAS)

    # Crop out the center 224x224 portion of the image
    left = (image.size[0] - 224) / 2
    top = (image.size[1] - 224) / 2
    right = (image.size[0] + 224) / 2
    bottom = (image.size[1] + 224) / 2
    image = image.crop((left, top, right, bottom))

    # Convert PIL image to NumPy array and normalize pixel values to [0, 1]
    np_image = np.array(image) / 255.0

    # Normalize the color channels using specified means and standard deviations
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means) / stds

    # Transpose the color channel to be the first dimension
    np_image = np_image.transpose((2, 0, 1))

    return np_image

    
        
def predict(image_path, model, topk):
    image = process_image(image_path)
    image = torch.tensor(image).unsqueeze(0).float()
    
    if args.gpu:
        device = 'cuda'
    else: 
        device = 'cpu'
    
    image = image.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(image)
        probabilities, indices = torch.topk(torch.softmax(output, dim=1), k=topk)
    
    return probabilities, indices

probabilities, indices = predict(args.path_to_image, model, args.top_k)


import json

with open(args.category_names, 'r') as f:
    flower_to_name = json.load(f)

indices = indices.squeeze().tolist()  # Convert tensor indices to a list of integers

names = [flower_to_name[str(index)] for index in indices]

final = []
for index in range(len(names)):
    final.append(names[index] + ": " + str(probabilities[0][index].item()))

print(final)
names = [flower_to_name[str(index)] for index in indices]

