import numpy as np
from sklearn.linear_model import LinearRegression
import tensorflow as tf

#定义一些变量和模型
x = tf.Variable(3.0)
y = x**2

# 使用 GradientTape 记录梯度
with tf.GradientTape() as tape:
    # 计算梯度
    grad = tape.gradient(y, x)

# 打印梯度
print("打印梯度 ：  "+str(grad))

print("Hello World")


arr = np.array([1, 2, 3, 4, 5])
print(arr)


X = [[1, 4], [2, 5], [3, 6]]
y = [8, 10, 12]
model = LinearRegression().fit(X, y)
print(model.predict([[4, 7]]))


