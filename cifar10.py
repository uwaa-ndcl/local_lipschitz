import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import my_config

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
net_name = 'cifar10'
plot_name = 'CIFAR-10 Net'

# this is only for computation and plotting
batch_size_l = 10000
batch_size_rand = 10**4
batch_size_sn = 100
batch_size_ball = 10000
#eps_min_net = .001
#eps_max_net = 2
eps_min = .001
eps_max = 2
eps_min_bisect = 4.1e-3 # new
eps_max_bisect = 4.5e-3 # new
step_size_grad = 1e-4
main_dir = 'data/cifar10/'

class MyNet(nn.Module):
    # https://keras.io/examples/cifar10_cnn/
    # trains to 75%/79% accuracy after 100 epochs using Adagrad
    # trains to 81%/84% accuracy after 500 epochs using Adagrad

    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3)
        self.conv2 = nn.Conv2d(32,32,3)
        self.conv3 = nn.Conv2d(32,64,3)
        self.conv4 = nn.Conv2d(64,64,3)

        self.pool = nn.MaxPool2d(2)

        self.do1 = nn.Dropout(.25)
        self.do2 = nn.Dropout(.25)
        self.do3 = nn.Dropout(.5)

        self.fc1 = nn.Linear(64*5*5,512)
        self.fc2 = nn.Linear(512,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.do1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.do2(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.do3(x)
        x = self.fc2(x)

        return x


def cifar10_mean_std():
    '''
    get mean and variance of a dataset 

    reference values: https://github.com/kuangliu/pytorch-cifar/issues/8
    '''

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
            root=cifar10_dir, transform=transform, train=True, download=True)
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


#cifar10_mean_std() # uncomment this to compute the values below
train_mean = (0.4914, 0.4822, 0.4465)
train_std = (0.2470, 0.2435, 0.2616)

# dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std),
])

# nominal input
filename = os.path.join(main_dir, 'dog4.png')
x0 = Image.open(filename)
x0 = transform_test(x0)
x0 = torch.unsqueeze(x0, 0)
x0 = x0.to(my_config.device)

def net():
    # load the checkpoint file
    net = MyNet()
    ckpt_file = os.path.join(main_dir, 'ckpt_499.ckpt')
    net.load_state_dict(torch.load(ckpt_file, map_location=my_config.device)['net'])

    # create a list of the network's layers
    #relu = torch.nn.functional.relu
    relu = torch.nn.ReLU(inplace=False)
    flatten = torch.nn.Flatten()
    net.layers = [net.conv1, relu,
                  net.conv2, relu,
                  net.pool,
                  net.conv3, relu,
                  net.conv4, relu,
                  net.pool,
                  flatten,
                  net.fc1, relu,
                  net.fc2] 

    return net
'''
# test an image
device = my_config.device
#net = VGG16()
net = MyNet()
net.to(device)
criterion = nn.CrossEntropyLoss()
def test_one():
    input = torch.randn(1,3,32,32).to(device)
    output = net(input)
    target = torch.randint(0,10,(1,)).to(device)  # a dummy target, for example
    loss = criterion(output, target)
    print(loss)

test_one()
'''
