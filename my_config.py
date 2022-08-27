import torch

# automatic
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# manual
#device = 'cuda'
device = 'cpu'
