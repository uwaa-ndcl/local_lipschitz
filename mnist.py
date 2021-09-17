import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import my_config

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
net_name = 'mnist'
plot_name = 'MNIST Net'

# this is only for computation and plotting
batch_size_l = 25000
batch_size_rand = 10**4
batch_size_sn = 100
batch_size_ball = 25000
eps_min = .001
eps_max = 7
step_size_grad = 1e-4
main_dir = 'data/mnist/'

# see:
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py
class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)

        self.maxpool = nn.MaxPool2d(2) 

        self.fc1 = nn.Linear(16*4*4,120) # note cifar10 is 16*5*5
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def mnist_mean_std():
    '''
    get mean and variance of a dataset 

    reference values: https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
    '''

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(
            root=mnist_dir, transform=transform, train=True, download=True)
    loader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=False, num_workers=1)

    # mean
    sum = 0.
    n_batches = 0.
    for batch_idx, (inputs, targets) in enumerate(loader):
        (batch_size, ch, h, w) = inputs.shape 
        sum += inputs.sum((0,2,3)) # sum over batch dimension
        n_batches += batch_size
    mean = sum/(n_batches*h*w)
    print('mean', mean)

    # std dev
    mean = mean[None,:,None,None]
    sum = 0.
    for batch_idx, (inputs, targets) in enumerate(loader):
        (batch_size, ch, h, w) = inputs.shape 
        sum += ((inputs-mean)**2).sum((0,2,3)) # sum over batch dimension
    denom = (n_batches-1)*h*w
    std = torch.sqrt(sum/denom)
    print('std', std)


#mnist_mean_std() # uncomment this to compute the values below
train_mean = (0.1307,)
train_std = (0.3081,)

# dataset
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std),
])

# nominal input
filename = os.path.join(main_dir, '8.png')
x0 = Image.open(filename)
x0 = transform_test(x0)
x0 = torch.unsqueeze(x0, 0)
x0 = x0.to(my_config.device)

def net():
    # load checkpoint file
    net = LeNet()
    ckpt_file = os.path.join(main_dir, 'ckpt_99.ckpt')
    net.load_state_dict(torch.load(ckpt_file)['net'])

    # create a list of the network's layers 
    #relu = torch.nn.functional.relu
    relu = torch.nn.ReLU(inplace=False)
    flatten = nn.Flatten()
    net.layers = [net.conv1, relu,
                  net.maxpool,
                  net.conv2, relu,
                  net.maxpool,
                  flatten,
                  net.fc1, relu,
                  net.fc2, relu,
                  net.fc3]

    return net
'''
# test an image
device = my_config.device
criterion = nn.CrossEntropyLoss()
net = LeNet()
net = net.to(device)
def test_one():
    input = torch.randn(1,1,28,28).to(device)
    output = net(input)
    target = torch.randint(0,10,(1,)).to(device)  # a dummy target, for example
    loss = criterion(output, target)
    print(loss)

test_one()
'''
