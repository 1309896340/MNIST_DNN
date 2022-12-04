import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 解决图像中的“-”负号的乱码问题

logtime = "2022_12_04_16_46_31"

fig = plt.figure(figsize=(12, 5))

loss, correct = np.load(f"log/loss_{logtime}.npy"), np.load(f"log/accuracy_{logtime}.npy")
n = len(loss) if len(loss) < len(correct) else len(correct)
loss = loss[:n]
correct = correct[:n]

t = np.arange(n)

downt = t[0:-1:300]
newt = np.arange(len(downt))
loss = loss[downt]
correct = correct[downt]

loss = loss / loss.max()

ax1 = plt.subplot(111)

l1 = ax1.plot(newt, loss, 'r', label="损失值")
plt.ylabel("损失值(归一化的)")
plt.yticks(np.arange(0, 1.1, 0.1))
plt.grid(visible=True, which='both')

ax2 = plt.twinx()

l2 = ax2.plot(newt, correct, 'b', label="准确率")
ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
plt.ylabel("准确率")
plt.xlim([0, 39])
plt.ylim([0, 1])

lns = l1 + l2
labs = [l.get_label() for l in lns]
plt.legend(lns, labs, loc="upper left")


plt.title("损失曲线/准确率曲线")
plt.xticks(range(40))
plt.xlabel("训练轮数(epoch)")
plt.xlim([0, 39])
plt.ylim([0, 1])

plt.show()

