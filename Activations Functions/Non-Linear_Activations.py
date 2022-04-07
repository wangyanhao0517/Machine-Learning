'''
Descripttion: 
version: 
Author: wyh
Date: 2022-04-06 17:17:56
LastEditors: wyh
LastEditTime: 2022-04-06 23:20:55
'''
import math

# Sigmoid激活函数
# 优点：
# 1. input越大，sigmoid输出越接近于1，反之越接近于0，类似于情感二分类任务；
# 2. 适用于那些预测某一项作为最后输出的概率的模型；
# 缺点：
# 1. 正负饱和区梯度接近0，会造成梯度消失（梯度弥散）；
# 2. 非0对称，数据非0均值，收敛速度下降；
# 3. 需要幂运算，运算成本高；

def Sigmoid(x):
    return 1/(1 + math.exp(-x))

def DerivateSigmoid(x):
    return Sigmoid(x) * (1 - Sigmoid(x))

# ************************************************

# Tanh激活函数
# 优点：
# 1. tanh 函数是0中心对称的，因此具有0均值的特点，收敛速度因为数据的0均值特点会比Sigmoid快；
# 2. 经常用作隐藏层的激活函数；
# 缺点：
# 1. 运算复杂，幂运算为主；
# 2. 其在正负饱和区的梯度都接近0，会造成梯度消失；

def Tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def DerivateTanh(x):
    return 1 - math.pow(Tanh(x), 2)

# ************************************************

# Relu激活函数
# 优点：
# 1. Relu不存在梯度饱和的问题，Tanh、Sigmoid都有梯度饱和的问题（当自变量进入某一区间之后，梯度变化会非常小；表现在图上就是函数曲线进入某些区域后，越来越趋近一条直线）
# 2. 改善了梯度消失问题；
# 3. 不需要指数运算，运算复杂度低；
# 4. 自变量小于0的部分神经元输出为0，造成网络的稀疏性，缓解过拟合问题；
# 缺点：
# 1. 均值大于0（对比Tanh），会存在偏移现象，导致网络收敛缓慢
# 2. 其在正负饱和区的梯度都接近0，会造成梯度消失；
# 3. 异常值敏感切且会出现神经元死亡的问题（对于稳定的网络，输入很大的X，产生很大的梯度，进一步导致偏置bias被更新为很小的负值，导致该神经元永远不会被激活）；
def Relu(x):
    return 0 if x<=0 else x

def DerivateRelu(x):
    return 0 if Relu(x) <= 0 else 1
