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
        # 激活函数实例化
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
        
        # 记录第一层输入
        self.input_net = input_x
        tmp = input_x
        for i in range(layer_n):
            # 正向传播(当前层连接权值矩阵 * 当前层输入 + 当前层偏置向量 ====当前层激活函数===> 当前层输出)
            # weights[i]    (batchsize, curNetNum, preNetNum)
            # tmp           (batchsize, preNetNum, 1)
            # biass[i]      (batchsize, curNetNum, 1)
            # 激活后的输出   (batchsize, curNetNum, 1)
            tmp = self.activators[i](self.weights[i] @ tmp + self.biass[i])

            # 进行dropout处理(注意: 最后一层不能dropout,概率为零导致交叉熵无穷大)
            if dropout and i != layer_n - 1:
                mask = torch.where(torch.rand(self.biass[i].size(), device=device) < dropratio, 1, 0)
                tmp = mask * tmp
            
            # 当前层输出值记录到net列表中
            self.net[i] = tmp

        # 返回最后一层的输出
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
        
        # 末层误差项delta计算
        delta = self.activators[k - 1].grad(self.net[k - 1], target)
        while k > 0:
            # 向前逐层迭代
            i = k - 1
            if i == 0:
                preNet = self.input_net
            else:
                preNet = self.net[i - 1]
            
            # 计算当前层连接权值矩阵和偏置向量的梯度
            dw = delta @ preNet.transpose(2, 1)
            db = delta

            # 更新前一层delta
            if i > 0:
                delta = self.activators[i - 1].grad(self.net[i - 1]) * (self.weights[i].transpose(1, 0) @ delta)

            # 使用adam优化并更新当前层权值梯度
            self.weights[i] = self.weights[i] - self.lr * self.optimizers[i][0].update(torch.mean(dw, axis=0))
            self.biass[i] = self.biass[i] - self.lr * self.optimizers[i][1].update(torch.mean(db, axis=0))

            k -= 1

    def test(self, t=getTime()):
        '''
        描述：
            在测试集上验证模型,输出混淆矩阵,输出评价参数,写入文件
        '''

        # 数据加载
        data_images, data_labels, n = load_data("MNIST/t10k-images-idx3-ubyte.gz", "MNIST/t10k-labels-idx1-ubyte.gz")
        data_images = data_images.to(device)
        data_labels = data_labels.to(device)
        
        # 预处理(展平、归一化、onehot编码)
        flatten_images = image_flatten(data_images)
        normalized_flatten_images = normalize(flatten_images)
        onehot_labels = onehot(data_labels)

        # 向前传播、输出概率分布映射为预测标签
        prd = self.forward(normalized_flatten_images, dropout=False)
        prd_labels = torch.argmax(prd, axis=1).squeeze()

        # 损失计算、准确度计算
        loss = cross_entropy_loss(prd, onehot_labels)
        accuracy = torch.sum(torch.where(data_labels == prd_labels, 1, 0)).item() / len(data_labels)

        # 数据分析(使用pandas.DataFrame存储)
        pdlabel = ['\'0\'', '\'1\'', '\'2\'', '\'3\'', '\'4\'', '\'5\'', '\'6\'', '\'7\'', '\'8\'', '\'9\'']
        cm = confusion_matrix(prd_labels.cpu().numpy(), data_labels.cpu().numpy())
        cm_data = pd.DataFrame(cm, index=pdlabel, columns=pdlabel)
        cm_ana = confuse_analysis(cm)
        cm_ana = pd.DataFrame(cm_ana, index=pdlabel, columns=["TP", "TN", "FP", "FN", "精确率", "召回率", "F值"])
        cm_ana[["TP", "TN", "FP", "FN"]] = cm_ana[["TP", "TN", "FP", "FN"]].astype(int)
        cm_ana["精确率"] = cm_ana["精确率"].map(lambda x: f"{x*100:.2f}%")
        cm_ana["召回率"] = cm_ana["召回率"].map(lambda x: f"{x*100:.2f}%")
        cm_ana["F值"] = cm_ana["F值"].map(lambda x: f"{x:.4f}")

        # 在终端输出
        print(f"输入测试样本{len(data_labels)}张")
        print(f"loss={loss}\naccuracy={accuracy*100:.2f}%")
        print(f"混淆矩阵:\n{cm_data}")
        print(f"真正,真负,假正,假负,精准率,召回率,F值:\n{cm_ana}")

        # 输出csv为日志
        cm_data.to_csv(f"log/ConfusionMatrix_{t}.csv", encoding="utf-8-sig")
        cm_ana.to_csv(f"log/ConfusionAnalysis_{t}.csv", encoding="utf-8-sig")
