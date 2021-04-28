# this script is to verify a batch normalization layer can be represented as an
# affine operator
import torch
from torchvision import models

# get batch norm layer
net = models.resnet18(pretrained=True)
net.eval()
batch_norm = net.bn1
#batch_norm = net.layer1[0].bn1
batch_norm.training = False

# array sizes
ch = batch_norm.weight.shape.numel()
h = 100
w = 200
print(ch, 'layers')

# parameters
W = batch_norm.weight
b = batch_norm.bias
E_x = batch_norm.running_mean
var_x = batch_norm.running_var
eps = batch_norm.eps
W = W[None,:,None,None]
B = b[None,:,None,None]
E_x = E_x[None,:,None,None]
var_x = var_x[None,:,None,None]

# input
X = torch.rand(1,ch,h,w)
#X = torch.ones(1,ch,h,w)
Y = batch_norm(X)

# function form 
#see: https://pytorch.org/docs/stable/nn.html#torch.nn.BatchNorm2d
Y2 = (X - E_x) / torch.sqrt(var_x + eps)
Y2 = Y2 * W + B
print('batch normalization v. function form error:', torch.norm(Y - Y2).item())

# affine form: A x + b
WW = W / torch.sqrt(var_x + eps)
BB = - E_x * W / torch.sqrt(var_x + eps) + B
Y3 = WW * X + BB
print('batch normalization v. affine form error:', torch.norm(Y - Y3).item())
