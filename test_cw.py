import torch
import torchvision
import torchvision.transforms as transforms
import torchattacks
from PIL import Image
import my_config

#import tiny as exp 
import mnist as exp 

transform_new = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.0,), (1.0,) ),
])


net = exp.net()
x0 = exp.x0
x0 = x0*exp.train_std[0] + exp.train_mean[0] # back to [0,1] interval

x0 = Image.open('/home/trevor/local_lipschitz/data/mnist/8.png')
x0 = transform_new(x0)
x0 = torch.unsqueeze(x0, 0)
x0 = x0.to(my_config.device)

y0 = net(x0)
ind_true = torch.topk(y0.flatten(), 1)[1].item()
n = 10
#labels = torch.randint(0,10,(n,))
labels = torch.linspace(0,n-1,n,dtype=torch.int64)
print(labels)
#labels[0] = 7
#labels = (2,)

attack = torchattacks.CW(net, c=1e-0, kappa=0, steps=1000, lr=0.01)
adv_images = attack(x0, labels)
Y = net(adv_images)
adv_ind = torch.topk(Y, 1)[1]
print(adv_ind)
for i in range(n):
    if adv_ind[i] != ind_true:
        print('ind:', i)
        print('true ind:', ind_true)
        print('attack ind:', adv_ind[i])
        print('diff:', torch.norm(adv_images[i,:,:,:] - x0))
