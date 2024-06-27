#coding:utf8

import torch
import numpy as np


'''
This codes is for numpy basic operation, it is also for torch basic operation.
Two ways to deal with matrix, one is numpy, another is torch, the first one is bascially for numpy, 
the second one is for torch, it is a special treatment.
'''

#numpy基本操作
x = np.array([[1,2,3],
              [4,5,6]])

# dimensionality i.e. rank
#The rank of a vector group is n, so the dimension of the resulting linear space of this vector group is n.
#Rank is a biased algebra and dimension is a biased geometry.
print("matrix rank   "+str(x.ndim))
# shape of an array
print("matrix shape  "+str(x.shape))
# total number of elements in a array
print(x.size)
# sum of all elements in an array
print(np.sum(x))
# sum of elements along a given axis
# axis=0 sum of all elements in a columm, i.e all data are added seperately in each column, or along the vertical direction of the horizon;
print("axis=0 sum "+str(np.sum(x, axis=0)))
# axis=1 sum of all elements in a column;
# sum of all elements in a raw, i.e all data are added seperately in each row, or along the vertical direction of the horizon;
print("axis=1 sum "+str(np.sum(x, axis=1)))
# reshape array
print("reshape array:"+str(np.reshape(x, (3,2))))
# square root of all elements in an array
print(np.sqrt(x))
# exponential of all elements in an array
print(np.exp(x))
# transpose of a matrix
print(x.transpose())
# flatten array
print(x.flatten())


#
print(np.zeros((3,4,5)))
# three 4*5-dimensional arrays with random values between 0 and 1
print(np.random.rand(3,4,5))
# three 4*5-dimensional arrays with random values between -1 and 1
print(np.random.randn(3,4,5))

#
x = np.random.rand(3,4,5)
x = torch.FloatTensor(x)
print("x.shape   "+str(x.shape))
print(x.shape)

print(torch.exp(x))
print(torch.sum(x, dim=0))
print(torch.sum(x, dim=1))
print(torch.sum(x, dim=2))



print(x.transpose(1, 0))
print(x.flatten())


