import os
from pathlib import Path

# get directory of this file
this_file = os.path.realpath(__file__)
this_dir = Path(this_file).parent
main_dir = this_dir.parent

# local directories
fig_dir = os.path.join(main_dir, 'fig/')
data_dir = os.path.join(main_dir, 'data/')
tiny_dir = os.path.join(data_dir, 'tiny/')
mnist_dir = os.path.join(data_dir, 'mnist/')
cifar10_dir = os.path.join(data_dir, 'cifar10/')
alexnet_dir = os.path.join(data_dir, 'alexnet/')
imagenet_dir = os.path.join(data_dir, 'imagenet/')
vgg16_dir = os.path.join(data_dir, 'vgg16/')
