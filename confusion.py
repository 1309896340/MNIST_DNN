import numpy as np

# 测试混淆矩阵参数计算


def confusion_matrix(predict_label, target_label):
    # 输入参数都是一维数组
    # 第i行表示实际值为i
    # 第j列表示预测值为j
    assert (isinstance(predict_label, np.ndarray))
    assert (isinstance(target_label, np.ndarray))
    predict_label = np.int0(predict_label)
    target_label = np.int0(target_label)
    n = 10
    mt = np.zeros([n, n])
    z = tuple(zip(predict_label, target_label))
    for a, b in z:
        mt[b, a] += 1
        # print(f"{b} -> {a}")
    return np.int0(mt)


def confuse_analysis(conf_mat):
    # 返回 TP TN FP FN
    assert (isinstance(conf_mat, np.ndarray))

    # 对的标签被正确识别为对的数量
    TP = np.diag(conf_mat)
    # 对的标签被误识别错的数量(如标签为3预测出来不是3的数量)
    FN = np.sum(conf_mat, axis=1) - TP
    # 错的标签被误识别对的数量(如预测出来是3但实际标签不是3的数量)
    FP = np.sum(conf_mat, axis=0) - TP
    # 错的标签被正确识别为错的数量(如预测不为3，实际也不是3的数量)
    TN = np.sum(conf_mat) - FN - FP - TP

    # 精确率
    precision = TP / (TP + FP)
    # 召回率
    recall = TP / (TP + FN)
    # F值
    F = (2 * precision * recall) / (precision + recall)

    return np.hstack([TP.reshape([-1, 1]), TN.reshape([-1, 1]), FP.reshape([-1, 1]), FN.reshape([-1, 1]), precision.reshape([-1, 1]), recall.reshape([-1, 1]), F.reshape([-1, 1])])


if __name__ == "__main__":
    # np.random.seed(43)
    tmp = np.arange(3)
    tmp = tmp.repeat(12)
    a = tmp.copy()
    np.random.shuffle(tmp)
    b = tmp.copy()
    print(f"实际标签a:{a}")
    print(f"预测标签b:{b}")
    res = confusion_matrix(a, b)
    cmparam = confuse_analysis(res)
    print(f"混淆矩阵:\n{res}")
    print(f"TP:{cmparam[0]} , TN:{cmparam[1]} , FP:{cmparam[2]} , FN:{cmparam[3]}")
