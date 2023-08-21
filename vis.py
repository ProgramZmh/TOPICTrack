import numpy as np
import matplotlib.pyplot as plt


# 生成示例数据
files = ["emb_cost.txt", "/remote-home/zhengyiyao/Deep-OC-SORT/deep_emb_cost.txt"]
for f in files:
    emb_cost = np.loadtxt(f)
    # print(emb_cost.shape)
    emb_cost = emb_cost[-1203:]

    x = np.arange(0, emb_cost.shape[0])
    y_mean = emb_cost[:, 0]
    y_std = emb_cost[:, 1]

    print("mean:", y_mean.mean())
    # 画均值和误差阴影图
    label = f.replace(".txt", "").split("/")[-1]
    plt.plot(x, y_mean, label=label)
    plt.fill_between(x, y_mean - y_std, y_mean + y_std,
                     alpha=0.5)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('appr sim')
plt.legend()
plt.grid(True)
plt.savefig("emb_cost.jpg")
plt.close()
