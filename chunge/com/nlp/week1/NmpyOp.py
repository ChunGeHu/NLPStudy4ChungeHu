#coding:utf8

import torch
import numpy as np



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



#torch
y = torch.FloatTensor([[1,2,3],
                       [4,5,0]])

# dimensionality i.e. rank
print("matrix dim   "+str(y.ndim))
print("matrix shape "+str(y.shape))
print(y.size)
print(np.sum(y))
print(np.sum(y, axis=0))
print(np.sum(y, axis=1))


print(np.reshape(y, (3,2)))
print(np.sqrt(y))
print(np.exp(x))
print(x.transpose())
print(x.flatten())

#
print("np.zeros"+str(np.zeros((3,4,5))))
# print(np.random.rand(3,4,5))
#
# x = np.random.rand(3,4,5)
x = torch.FloatTensor(x)
print(x.shape)
print(torch.exp(x))
print(torch.sum(x, dim=0))
print(torch.sum(x, dim=1))
print(x.transpose(1, 0))
print(x.flatten())

