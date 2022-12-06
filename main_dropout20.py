from abc import abstractmethod
import pickle
import numpy as np

import torch
from model import MyModel
from activator import *
from confusion import *
from utils import *

import atexit


device = torch.device("cuda:0")


t = None
m = None

loss_log = []
validloss_log = []
accuracy_log = []


def log_output():
    '''
    描述：
        导出损失值和准确率
    '''
    np.save(f"log/loss_{t}.npy", np.array(loss_log))
    np.save(f"log/validloss_{t}.npy", np.array(validloss_log))
    np.save(f"log/accuracy_{t}.npy", np.array(accuracy_log))
    print(f"训练结束,导出日志为 {t}")
    
    # 在测试集上验证模型
    m.test(t)
    # 导出训练好的模型
    with open(f"log/{t}.model", "wb") as fp:
        pickle.dump(m, fp)


if __name__ == '__main__':
    t = getTime()

    # 程序异常退出时掉用log_output以输出日志
    atexit.register(log_output)

    # 参数集中管理
    param = {"lr": 0.00002, "epoch": 40, "batchsize": 300, "train_rate": 0.8, "dropratio": 0.8}

    # 加载数据集
    data_images, data_labels, n = load_data(fimg="MNIST/train-images-idx3-ubyte.gz", flab="MNIST/train-labels-idx1-ubyte.gz")

    # 将待训练数据转移到GPU，调用GPU运算
    data_images = data_images.to(device)
    data_labels = data_labels.to(device)

    # 将(batchsize,28,28)展平为(batchsize,784,1)
    data_images = image_flatten(data_images)

    # 分割训练集(train)和验证集(valid)
    train_images, train_labels, valid_images, valid_labels, train_n, valid_n = \
    split_train_test(data_images, data_labels, train_rate=param['train_rate'])

    # 像素值归一化
    train_images = normalize(train_images)
    valid_images = normalize(valid_images)

    # 创建模型，设置学习率，添加网络层
    m = MyModel(lr=param['lr'])
    m.addLayer(784, 1024, activator=leakyRelu)
    m.addLayer(1024, 1024, activator=leakyRelu)
    m.addLayer(1024, 512, activator=leakyRelu)
    m.addLayer(512, 256, activator=sigmoid)
    m.addLayer(256, 10, activator=softmax_crosentropy)

    # 设置训练轮数epoch和一个batch的样本数量
    epoch = param['epoch']
    batchsize = param['batchsize']

    batchnum = train_n // batchsize
    for i in range(epoch):
        # 每一轮开始前打乱样本次序
        train_images, train_labels = data_shuffle(train_images, train_labels)
        print(f"======================开始第{i+1}轮===========================")
        for k in range(batchnum):
            # 取出一个batch的图片和标签
            batch_imags = train_images[k * batchsize : (k + 1) * batchsize]
            batch_labels = train_labels[k * batchsize : (k + 1) * batchsize]

            # 输入样本，向前传播，输出形状为(batchsize,10,1)的张量
            output = m.forward(batch_imags, dropout=True, dropratio=param['dropratio'])

            # 将真实标签onthot处理，得到(batchsize,10,1)的张量
            onehot_batch_labels = onehot(batch_labels)

            # 评估一个batch上的损失值，输出日志
            loss = cross_entropy_loss(output, onehot_batch_labels)
            loss_log.append(loss)

            # 根据标签值计算反向传播
            m.backward(onehot_batch_labels)

            # 在验证集上进行测试(向前传播、输出概率分布映射为预测标签、标签onehot编码)
            prd = m.forward(valid_images)
            prd_labels = torch.argmax(prd, axis=1).squeeze()
            onehot_valid_labels = onehot(valid_labels)

            # 计算损失值，输出日志
            valid_loss = cross_entropy_loss(prd, onehot_valid_labels)
            validloss_log.append(valid_loss)

            # 计算准确率，输出日志
            accuracy = torch.sum(torch.where(valid_labels == prd_labels, 1, 0)).item() / len(valid_labels)
            accuracy_log.append(accuracy)

            if k % 40 == 0:
                print(f"batch loss={loss}")
                print(f"valid loss={valid_loss}")
                print(f"valid accuracy={accuracy*100:.2f}%")
