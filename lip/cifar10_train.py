import os.path
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import dirs
import my_config
import cifar10

device = my_config.device
cifar10_dir = dirs.cifar10_dir

# datasets
n_workers = 8
batch_size = 128
trainset = torchvision.datasets.CIFAR10(
    root=cifar10_dir, train=True, download=True, transform=cifar10.transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=n_workers)

testset = torchvision.datasets.CIFAR10(
    root=cifar10_dir, train=False, download=True, transform=cifar10.transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

# network
net = cifar10.MyNet()
net = net.to(device)

# loss
lr = .1 # learning rate
#criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
#optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)
optimizer = torch.optim.Adagrad(net.parameters())
#optimizer = torch.optim.Adadelta(net.parameters())
#optimizer = torch.optim.RMSprop(net.parameters())

# iterate over epochs
n_epochs = 500
for i in range(n_epochs):
    print('\nepoch:     ', i)

    # train
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # error
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('train loss:', train_loss)
    correct_pct = 100*(correct/total)
    print('correct %: ', correct_pct)

    # test
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('test loss: ', test_loss)
        correct_pct = 100*(correct/total)
        print('correct %: ', correct_pct)

# save checkpoint
state = {'net': net.state_dict(), 'correct_pct': correct_pct, 'epoch': i}
ckpt_file = os.path.join(cifar10_dir, 'ckpt_' + str(i) + '.ckpt')
torch.save(state, ckpt_file)
