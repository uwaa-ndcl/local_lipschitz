import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

import my_config

net_name = 'vgg16'
plot_name = 'VGG-16'

# this is only for computation and plotting
#batch_size_l = 200
batch_size_l = 180
batch_size_rand = 200
batch_size_sn = 100
batch_size_ball = 100

eps_min = .001
eps_max = 3

#eps_min_bisect = 1.4e-7 # old
#eps_max_bisect = 1.6e-7 # old
eps_min_bisect = 1.7e-8 # new
eps_max_bisect = 1.9e-8 # new
step_size_grad = 1e-4

main_dir = 'data/vgg16/'
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

    net = models.vgg16(pretrained=True)
    net = net.eval()

    # create a list of layers of the network
    layers = []
    relu = nn.ReLU(inplace=False)
    flatten = nn.Flatten()

    # list of all network functions
    #funs = nn.Sequential((*list(net.features)+[net.avgpool]+[flatten]+list(net.classifier)))
    layers = list(net.features)+[net.avgpool]+[flatten]+list(net.classifier)
    net.layers = layers

    return net
