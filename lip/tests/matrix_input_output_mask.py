import torch

# setup
m = 4
n = 3
A = torch.rand(m,n)
zero_input_inds = torch.BoolTensor([0,1,1])
zero_output_inds = torch.BoolTensor([1,0,0,1])
#zero_input_inds = torch.BoolTensor(torch.rand(n)<.5)
#zero_output_inds = torch.BoolTensor(torch.rand(m)<.5)
one_input_inds = ~zero_input_inds
one_output_inds = ~zero_output_inds
R_i = torch.diag(one_input_inds.to(torch.float))
R_o = torch.diag(one_output_inds.to(torch.float))
A_new = R_o @ A @ R_i

# torch SVD
u,sss,v = torch.svd(A_new)
svd_spec = torch.max(sss)
print('svd     ', svd_spec.item())

# power iteration
v = torch.rand(n)
for i in range(1000):
    v = A_new.T @ A_new @ v
    v = torch.nn.functional.normalize(v, dim=0)
w = A_new @ v
pow_spec = torch.norm(w)
print('pow     ', pow_spec.item())

# power iteration mask
v = torch.rand(n)
v[zero_input_inds] = 0
for i in range(1000):
    w = A @ v
    w[zero_output_inds] = 0
    v = A.T @ w
    v[zero_input_inds] = 0
    v = torch.nn.functional.normalize(v, dim=0)
w = A @ v
w[zero_output_inds] = 0
pow_spec_mask = torch.norm(w)
print('pow mask', pow_spec_mask.item())

print('error', torch.norm(svd_spec - pow_spec))
print('error mask', torch.norm(svd_spec - pow_spec_mask))
