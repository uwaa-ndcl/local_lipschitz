import os.path
import torch
import torch.nn as nn
import torchvision

import my_config
import mnist

device = my_config.device
mnist_dir = exp.main_dir

# data sets
n_workers = 8
batch_size = 128
trainset = torchvision.datasets.MNIST(
    root=mnist_dir, train=True, download=True, transform=mnist.transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=n_workers)

testset = torchvision.datasets.MNIST(
    root=mnist_dir, train=False, download=True, transform=mnist.transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

# network
net = mnist.LeNet()
net = net.to(device)

# loss
lr = .1 # learning rate
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)
#optimizer = torch.optim.Adadelta(net.parameters(), lr=lr)

# iterate over epochs
n_epochs = 100
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
ckpt_file = os.path.join(mnist_dir, 'ckpt_' + str(i) + '.ckpt')
torch.save(state, ckpt_file)
