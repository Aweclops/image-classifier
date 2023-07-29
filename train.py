import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict
import torch.nn.functional as F
from PIL import Image
import glob,os
import numpy

parser = argparse.ArgumentParser()

parser.add_argument('data_directory',type=str)
parser.add_argument('--save_dir',type=str,required=False, default='')
parser.add_argument('--arch', type=str, required=False,default='vgg11')
parser.add_argument('--learning_rate', type=float, required=False, default=0.01)
parser.add_argument('--hidden_units', type=int, required=False,default=2048)
parser.add_argument('--epochs', type=int, required=False, default=15)
parser.add_argument('--gpu', action='store_true', default=False,required=False)
args = parser.parse_args()



# define transforms
data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])

train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor()])


train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)

test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)


trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=32,shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset,batch_size=32, shuffle=True)

import json

#with open('cat_to_name.json', 'r') as f:
#    cat_to_name = json.load(f)

# define model


model = getattr(models, args.arch)(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
    
# define classifier
model.classifier = nn.Sequential(nn.Linear(25088,args.hidden_units),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(args.hidden_units,102),
                                nn.LogSoftmax(dim=1))

# move model to gpu

if args.gpu == True:    
    device = 'cuda'
else: 
    device = 'cpu' 
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(),lr=args.learning_rate)
model.to(device)

# train model

epochs = args.epochs
running_loss = 0
steps = 1
accuracy = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for epoch in range(epochs):
    
    for images, labels in trainloader:
        model.train()
        images,labels = (images.to(device)).long(), labels.to(device)

        log_ps = model.forward(images.float())


        loss = criterion(log_ps, labels)
        
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        steps+=1
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        if steps % 10==0:
            print(f"Epoch {epoch+1}/{epochs}.. "
            f"Test accuracy: {accuracy/len(trainloader):.3f}")
        

print('finished')


# save checkpoint

checkpoint = {'input_size': 25088,
              'output_size': 102,
              'optimizer': optimizer.state_dict(),
              'class_to_idx': train_dataset.class_to_idx,
              'hidden_layer_size': args.hidden_units,
              'state_dict': model.state_dict()}

torch.save(checkpoint, args.save_dir + '/image-classifier.pth')


