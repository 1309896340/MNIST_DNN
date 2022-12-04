import torch
device = torch.device("cuda:0")

class AdamOptimizer:

    def __init__(self, grad_size, alpha=0.9, rho=0.999, sigma=1e-7):
        '''
        描述：
            adam优化器初始化
        参数：
            grad_size: list, tuple
                待优化参数的梯度矩阵形状
            alpha: float
                动量因子
            rho: float
                衰减因子
            sigma: float
                避免除以0的小数值
        示例：
            adam = AdamOptimizer([10, 1024])
            adam = AdamOptimizer([10, 1024], alpha=0.85)
            adam = AdamOptimizer((10, 1024), alpha=0.85, rho=0.9999)
            adam = AdamOptimizer((10, 1024), alpha=0.85, rho=0.9999, sigma=1e-10)
        '''
        self.m = torch.zeros(grad_size, device=device)
        self.v = torch.zeros(grad_size, device=device)

        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho

        self.t = 1

    def reset(self):
        '''
        描述：
            重置内部计数器
        '''
        self.t = 0

    def update(self, grad):
        '''
        描述：
            更新梯度
        参数：
            grad: torch.Tensor
                梯度矩阵，形状在初始化时指定
        返回值：
            torch.Tensor
                优化后的梯度矩阵，形状在初始化时指定
        '''
        self.m = self.alpha * self.m + (1 - self.alpha) * grad
        self.v = self.rho * self.v + (1 - self.rho) * (grad**2)
        nor_v = self.m / (1 - self.alpha**self.t)
        nor_r = self.v / (1 - self.rho**self.t)
        ngrad = nor_v / (torch.sqrt(nor_r) + self.sigma)

        self.t += 1
        return ngrad
