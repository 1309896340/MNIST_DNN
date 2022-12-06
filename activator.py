import torch
import numpy as np
from abc import abstractmethod


class activator:
    '''
        激活函数
    '''

    def __call__(self, x):
        return self.activate(x)

    @abstractmethod
    def parameter_init_scale(self, M1, M2):
        '''
        描述：
            对参数进行高斯分布随机初始化
            对不同的激活函数,不同的sigma
        参数：
            M1: int
                当前层神经元数量
            M2: int
                上一层神经元数量
        返回值：
            float
                该层高斯分布的标准差系数
        '''
        pass

    @abstractmethod
    def activate(self, x):
        '''
        描述：
            对线性运算的输出值进行激活
        '''
        pass

    @abstractmethod
    def grad(self, y):
        '''
        描述：
            根据激活后的输出值计算梯度
        '''
        pass


class relu(activator):
    '''
        relu激活函数
    '''

    def __init__(self):
        super().__init__()

    def parameter_init_scale(self, M1, M2=1):
        return np.sqrt(2 / M1)

    def activate(self, x):
        return torch.where(x > 0, x, 0)

    def grad(self, y):
        return torch.where(y > 0, 1, 0)


class leakyRelu(activator):
    '''
        leakyRelu激活函数
    '''

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def parameter_init_scale(self, M1, M2=1):
        return np.sqrt(2 / M1)

    def activate(self, x):
        return torch.where(x > 0, x, self.alpha * x)

    def grad(self, y):
        return torch.where(y > 0, 1, self.alpha)


class sigmoid(activator):
    '''
        sigmoid激活函数
    '''

    def __init__(self):
        super().__init__()

    def parameter_init_scale(self, M1, M2=1):
        return 1

    def activate(self, x):
        return 1 / (1 + torch.exp(-x))

    def grad(self, y):
        return y * (1 - y)


class tanh(activator):
    '''
        tanh激活函数
    '''

    def __init__(self):
        super().__init__()

    def parameter_init_scale(self, M1, M2=1):
        return np.sqrt(2 / (M1 + M2))

    def activate(self, x):
        ez = torch.exp(x)
        enz = 1 / ez
        res = (ez - enz) / (ez + enz)
        return res

    def grad(self, y):
        return 1 - y**2


class softmax_crosentropy(activator):

    def __init__(self):
        super().__init__()

    def parameter_init_scale(self, M1, M2=1):
        return 1

    def activate(self, x):
        y = torch.exp(x)
        return y / torch.sum(y, axis=1).unsqueeze(2)

    def grad(self, y, onehot_label):
        '''
        描述：
            使用交叉熵的混合梯度
        参数：
            y: torch.Tensor
                形状为(batchsize, 10, 1)
            onehot_label: torch.Tensor
                形状为(batchsize, 10, 1)
                    真实标签的onehot编码
        '''
        return y - onehot_label

