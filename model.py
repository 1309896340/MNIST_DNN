import torch
import pandas as pd

from optimizer import AdamOptimizer
from activator import *
from utils import *
from confusion import *

device = torch.device("cuda:0")


class MyModel(object):
    weights = None
    biass = None
    net = None
    input_net = None
    activators = None
    lr = None
    optimizers = None

    sigma = 1e-10

    def __init__(self, lr=0.001):
        self.weights = []
        self.biass = []
        self.net = []
        self.activators = []
        self.optimizers = []
        self.lr = lr

    def addLayer(self, input_features, output_features, activator=relu):
        # 实例化激活函数
        actvt = activator()
        # 参数初始化
        pre_nerv_n = 1 if len(self.net) == 0 else len(self.net[-1])
        cur_nerv_n = output_features
        sigma_scale = actvt.parameter_init_scale(cur_nerv_n, pre_nerv_n)
        weight = torch.randn([output_features, input_features], device=device) * sigma_scale
        bias = torch.randn([output_features, 1], device=device) * sigma_scale

        self.weights.append(weight)
        self.biass.append(bias)
        self.net.append([])
        self.activators.append(actvt)
        self.optimizers.append((AdamOptimizer(weight.size()), AdamOptimizer(bias.size())))

    def forward(self, input_x, dropout=False, dropratio=0.9):
        '''
        描述：
            模型向前传播
        参数：
            input_x: torch.Tensor
                形状为(图片数, 像素数, 1)
            dropout: bool
                是否开启dropout
            dropratio: float
                取值范围(0, 1)
                大于这个值的节点被屏蔽
        返回值：
            torch.Tensor
                形状为(batchsize, 输出神经元数, 1)
        '''
        layer_n = len(self.weights)
        self.input_net = input_x
        tmp = input_x
        for i in range(layer_n):
            tmp = self.activators[i](self.weights[i] @ tmp + self.biass[i])
            # dropout处理(注意: 最后一层不能dropout,概率为零导致交叉熵无穷大)
            if dropout and i != layer_n - 1:
                mask = torch.where(torch.rand(self.biass[i].size(), device=device) < dropratio, 1, 0)
                tmp = mask * tmp
            self.net[i] = tmp
        return tmp

    def backward(self, target):
        '''
        描述：
            模型的反向传播
        参数：
            target: torch.Tensor
                形状为(batchsize, 10, 1)
                整个batch的onehot标签
        '''
        k = len(self.net)
        delta = self.activators[k - 1].grad(self.net[k - 1], target)
        while k > 0:
            i = k - 1

            if i == 0:
                preNet = self.input_net
            else:
                preNet = self.net[i - 1]
            dw = delta @ preNet.transpose(2, 1)
            db = delta

            # 更新前一层的delta
            if i > 0:
                delta = self.activators[i - 1].grad(self.net[i - 1]) * (self.weights[i].transpose(1, 0) @ delta)

            self.weights[i] = self.weights[i] - self.lr * self.optimizers[i][0].update(torch.mean(dw, axis=0))
            self.biass[i] = self.biass[i] - self.lr * self.optimizers[i][1].update(torch.mean(db, axis=0))

            k -= 1

    def test(self, t=getTime()):
        '''
        描述：
            在测试集上验证模型,输出混淆矩阵,输出评价参数,写入文件
        '''
        data_images, data_labels, n = load_data("MNIST/t10k-images-idx3-ubyte.gz", "MNIST/t10k-labels-idx1-ubyte.gz")
        data_images = data_images.to(device)
        data_labels = data_labels.to(device)

        flatten_images = image_flatten(data_images)
        normalized_flatten_images = normalize(flatten_images)
        prd = self.forward(normalized_flatten_images, dropout=False)
        onehot_labels = onehot(data_labels)
        loss = cross_entropy_loss(prd, onehot_labels)

        prd_labels = torch.argmax(prd, axis=1).squeeze()
        accuracy = torch.sum(torch.where(data_labels == prd_labels, 1, 0)).item() / len(data_labels)

        pdlabel = ['\'0\'', '\'1\'', '\'2\'', '\'3\'', '\'4\'', '\'5\'', '\'6\'', '\'7\'', '\'8\'', '\'9\'']
        cm = confusion_matrix(prd_labels.cpu().numpy(), data_labels.cpu().numpy())
        cm_data = pd.DataFrame(cm, index=pdlabel, columns=pdlabel)
        cm_ana = confuse_analysis(cm)
        cm_ana = pd.DataFrame(cm_ana, index=pdlabel, columns=["TP", "TN", "FP", "FN", "精确率", "召回率", "F值"])
        cm_ana[["TP", "TN", "FP", "FN"]] = cm_ana[["TP", "TN", "FP", "FN"]].astype(int)
        cm_ana["精确率"] = cm_ana["精确率"].map(lambda x: f"{x*100:.2f}%")
        cm_ana["召回率"] = cm_ana["召回率"].map(lambda x: f"{x*100:.2f}%")
        cm_ana["F值"] = cm_ana["F值"].map(lambda x: f"{x:.4f}")

        print(f"输入测试样本{len(data_labels)}张")
        print(f"loss={loss}\naccuracy={accuracy*100:.2f}%")
        print(f"混淆矩阵:\n{cm_data}")
        print(f"真正,真负,假正,假负,精准率,召回率,F值:\n{cm_ana}")

        cm_data.to_csv(f"log/ConfusionMatrix_{t}.csv", encoding="utf-8-sig")
        cm_ana.to_csv(f"log/ConfusionAnalysis_{t}.csv", encoding="utf-8-sig")
