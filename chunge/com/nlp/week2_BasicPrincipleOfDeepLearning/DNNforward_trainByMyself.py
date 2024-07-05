#coding:utf8

import torch
import torch.nn as nn
import numpy as np

"""
numpy手动实现模拟一个线性层
"""

#搭建一个2层的神经网络模型
#每层都是线性层
class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1) #w：3 * 5
        self.layer2 = nn.Linear(hidden_size1, hidden_size2) # 5 * 2
    def forward(self, x):
        x = self.layer1(x)   #shape: (batch_size, input_size) -> (batch_size, hidden_size1) 
        y_pred = self.layer2(x) #shape: (batch_size, hidden_size1) -> (batch_size, hidden_size2) 
        return y_pred


# Y = W * X +B
#随便准备一个网络输入
x = np.array([[3.1, 1.3, 1.2],
              [2.1, 1.3, 13]])
#建立torch模型
# 这里的size可以这么理解： 输入 2 * 3，hidden_layer 3 * 5, hidden_layer2 5*2;
torch_model = TorchModel(3, 5, 2)

#使用torch模型做预测
torch_x = torch.FloatTensor(x)
y_pred = torch_model.forward(torch_x)
print("torch模型预测结果：", y_pred)