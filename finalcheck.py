import pickle
import numpy as np
import torch

from confusion import *
from model import MyModel
from utils import *
import pandas as pd

device = torch.device("cuda:0")
np.set_printoptions(threshold=10000, linewidth=200)


def test():

    logtime = "2022_12_05_21_22_08"

    with open(f"log/{logtime}.model", "rb") as fp:

        m = pickle.load(fp)
        assert (isinstance(m, MyModel))
        data_images, data_labels, n = load_data("MNIST/t10k-images-idx3-ubyte.gz", "MNIST/t10k-labels-idx1-ubyte.gz")
        data_images = data_images.to(device)
        data_labels = data_labels.to(device)

        flatten_images = image_flatten(data_images)
        normalized_flatten_images = normalize(flatten_images)
        prd = m.forward(normalized_flatten_images, dropout=False)
        onehot_labels = onehot(data_labels)
        loss = cross_entropy_loss(prd, onehot_labels)

        prd_labels = torch.argmax(prd, axis=1).squeeze()
        accuracy = torch.sum(torch.where(data_labels == prd_labels, 1, 0)).item() / len(data_labels)

        cm = confusion_matrix(prd_labels.cpu().numpy(), data_labels.cpu().numpy())
        cm_data = pd.DataFrame(cm, index=['\'0\'', '\'1\'', '\'2\'', '\'3\'', '\'4\'', '\'5\'', '\'6\'', '\'7\'', '\'8\'', '\'9\''], columns=['\'0\'', '\'1\'', '\'2\'', '\'3\'', '\'4\'', '\'5\'', '\'6\'', '\'7\'', '\'8\'', '\'9\''])
        cm_ana = confuse_analysis(cm)
        cm_ana = pd.DataFrame(cm_ana, index=['\'0\'', '\'1\'', '\'2\'', '\'3\'', '\'4\'', '\'5\'', '\'6\'', '\'7\'', '\'8\'', '\'9\''], columns=["TP", "TN", "FP", "FN", "精确率", "召回率","F值"])
        cm_ana[["TP", "TN", "FP", "FN"]] = cm_ana[["TP", "TN", "FP", "FN"]].astype(int)
        cm_ana["精确率"] = cm_ana["精确率"].map(lambda x: f"{x*100:.2f}%")
        cm_ana["召回率"] = cm_ana["召回率"].map(lambda x: f"{x*100:.2f}%")
        cm_ana["F值"] = cm_ana["F值"].map(lambda x: f"{x:.4f}")
        

        print(f"输入测试样本{len(data_labels)}张")
        print(f"loss={loss}\naccuracy={accuracy*100:.2f}%")
        print(f"混淆矩阵:\n{cm_data}")
        print(f"真正,真负,假正,假负,精准率,召回率,F值:\n{cm_ana}")

        cm_data.to_csv(f"log/ConfusionMatrix_{logtime}.csv", encoding="utf-8-sig")
        cm_ana.to_csv(f"log/ConfusionAnalysis_{logtime}.csv", encoding="utf-8-sig")


test()
