import gzip
import struct
import torch
from torch import Tensor
import numpy as np
import time

device = torch.device("cuda:0")


def getTime():
    '''
    描述：
        保存开始训练的时间,作为导出日志文件名
    '''
    return time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())


def load_data(fimg, flab):
    '''
    描述：
        加载训练数据
    参数：
        fimg: str
            图片数据的相对目录文件名
        flab: str
            标签数据的相对目录文件名
    返回值：
        (图片, 标签, 数量)
            图片: torch.Tensor
                形状为(样本数, 28, 28)
            标签: torch.Tensor
                形状为(样本数)
            数量: int
                样本数
    '''
    data_images = []
    data_labels = []
    with gzip.open(fimg) as fp:
        # 魔法数
        magicnbr = struct.unpack(">i", fp.read(4))
        # 样本图片数
        n, = struct.unpack(">i", fp.read(4))
        # 图片行数
        row, = struct.unpack(">i", fp.read(4))
        # 图片列数
        col, = struct.unpack(">i", fp.read(4))
        pixel_nbr_of_image = row * col
        for _ in range(n):
            pixels = struct.unpack(f">{pixel_nbr_of_image}B", fp.read(pixel_nbr_of_image))
            img = np.array(pixels, dtype=np.uint8).reshape([row, col])
            data_images.append(img)

    with gzip.open(flab) as fp:
        # 魔法数
        magicnbr = struct.unpack(">i", fp.read(4))
        # 样本标签数
        n, = struct.unpack(">i", fp.read(4))
        for _ in range(n):
            label, = struct.unpack(f">B", fp.read(1))
            data_labels.append(label)
    return Tensor(np.array(data_images)), Tensor(np.array(data_labels)), n


# 分割训练和验证集，默认8:2
def split_train_test(images, labels, train_rate=0.8):
    '''
    描述：
        分割训练集和验证集
    参数：
        images: torch.Tensor
            形状为(图片数, ...)的
        labels: torch.Tensor
            形状为(图片数)
    返回值：
        (训练集图片, 训练集标签, 验证集图片, 验证集标签, 训练集样本数, 验证集样本数)
            训练集图片, 验证集图片: torch.Tensor
                形状为(样本数, ...)
            训练集标签, 训练集样本数: torch.Tensor
                形状为(样本数)
            训练集样本数, 验证集样本数
                int
    '''
    n = len(labels)
    train_n = int(0.5 + n * train_rate)
    valid_n = n - train_n
    return images[:train_n, ...], labels[:train_n], images[train_n:, ...], labels[train_n:], train_n, valid_n


def image_flatten(imgs):
    '''
    描述：
        将图片(批量)展平到一维
    参数：
        imgs: torch.Tensor
            形状为(图片数, 行数, 列数)
    返回值：
        torch.Tensor
            形状为(图片数, 像素数, 1)
            像素数=行数*列数
    '''
    batchsize = imgs.size(0)
    return imgs.reshape([batchsize, -1, 1])


def data_shuffle(images, labels):
    '''
    描述：
        对图片和标签进行同步乱序
    参数：
        imags: torch.Tensor
            形状为(图片数, ...)
        labels: torch.Tensor
            形状为(图片数)
    返回值：
        (imags, labels)
            imags: torch.Tensor
                形状为(图片数, ...)
            labels: torch.Tensor
                形状为(图片数)
    示例：
        new_imgs, new_labels = data_shuffle(old_imgs, old_labels)
    '''
    n = len(labels)
    idxs = torch.randperm(n, device=device)
    return images[idxs], labels[idxs].int()


def normalize(imgs):
    '''
    描述：
        对图片(批量)进行归一化
    参数：
        imgs: torch.Tensor
        形状为(图片数, ...)的
    '''
    return imgs / 255


def center_normalize(imgs):
    '''
    描述：
        对图片(批量)进行归一化
    参数：
        imgs: torch.Tensor
        形状为(图片数, ...)的
    '''
    return ((imgs / 255) - 0.5) * 2


def gauss_normalize(imgs):
    '''
    描述：
        对图片(批量)进行归一化
    参数：
        imgs: torch.Tensor
            形状为(batchsize, ...)
    返回值：
        torch.Tensor
            形状为(batchsize, ...)
    '''
    return (imgs - torch.mean(imgs, axis=1).unsqueeze(1)) / torch.std(imgs, axis=1).unsqueeze(1)


# 进行onehot编码(平滑处理)
labelmap = torch.eye(10, device=device) * 0.998 + 0.001

def onehot(labels):
    '''
    描述：
        对标签进行onehot编码
    参数：
        labels: torch.Tensor
            形状为(batchsize)
    返回值：
        torch.Tensor
            形状为(batchsize, 10, 1)
    '''
    idxs = labels.long()
    return labelmap[idxs].unsqueeze(axis=2)




def cross_entropy_loss(x, y):
    '''
    描述：
        交叉熵损失函数，仅用于评估风险
    参数：
        x: torch.Tensor
            形状为(batchsize, 10, 1)
        y: torch.Tensor
            形状为(batchsize, 10, 1)
    返回值：
        float
            交叉熵损失的平均值
    '''
    x = torch.where(x == 0, 1e-20, x)
    return -torch.sum(y * torch.log(x)).item() / x.size(0)
