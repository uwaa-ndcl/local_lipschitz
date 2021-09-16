import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

import my_config

net_name = 'alexnet'
plot_name = 'AlexNet'

# this is only for computation and plotting
batch_size_l = 2000
batch_size_rand = 10**3
batch_size_sn = 100
batch_size_ball = 2000
eps_min = .001
eps_max = 3
step_size_grad = 1e-3
main_dir = 'data/alexnet/'
imagenet_dir = 'data/imagenet/'

# function to transform imagenet images
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])])

# nominal input
filename = os.path.join(imagenet_dir, 'toucan.png')
x0 = Image.open(filename)
x0 = transform(x0)
x0 = torch.unsqueeze(x0, 0)
x0 = x0.to(my_config.device)

# get class names
classes_file = os.path.join(imagenet_dir, 'classes.txt')
with open(classes_file) as f:
    classes = [line.rstrip() for line in f]
    classes = [line.split(': ', 1)[1] for line in classes]

def net():

    net = models.alexnet(pretrained=True)
    net = net.eval()

    # create a list of layers of the network
    layers = []
    relu = nn.ReLU(inplace=False)
    flatten = nn.Flatten()

    # list of all network functions
    #funs = nn.Sequential((*list(net.features)+[net.avgpool]+[flatten]+list(net.classifier)))
    layers = list(net.features)+[net.avgpool]+[flatten]+list(net.classifier)
    '''
    for i, layer in enumerate(fun_list):
        if isinstance(layer, nn.Dropout):
            del fun_list[i]
    '''

    net.layers = layers

    return net
