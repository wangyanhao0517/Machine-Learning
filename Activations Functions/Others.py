# 线性激活函数
# 函数：f(X) = X
# 缺点：
# 1. BackPropagation 失效，因为求倒数之后是常数；
# 2. 整个NN相当于对于第一层进行了多次的线性变换，无论有多少层，output层仍然是第一层的线性函数

# 二次阶跃激活函数
# 函数：f(x) = 1 if x>=0 else 0
# 缺点：
# 1. 反向传播障碍，其本身是常数，无法反向传播；
# 2. 无法进行多分类任务，只有二值特征，无法提供多个分类值的输出；