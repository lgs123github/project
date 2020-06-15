import torch
x=torch.tensor([[[[0,1,2,3],[3,4,5,6],[6,7,8,9]],[[1,2,3,4],[4,5,6,7],[7,8,9,10]],[[0,1,2,3],[3,4,5,6],[6,7,8,9]],[[1,2,3,4],[4,5,6,7],[7,8,9,10]]]])
print(x.shape)
mask=x[...,0]>0#[2,3,4]最后一个维度大于0作为掩码，掩码形状【2,3】
print(mask)
print(mask.shape)
print(mask.nonzero())
print(mask.nonzero().shape)
exit()
y=x[mask]#【2,3,4】求最后一个维度大于0的框，形状【框的个数，最后一个维度的数字】
print(mask)
print(y)
print(mask.shape,y.shape)
print(x[...,0])



