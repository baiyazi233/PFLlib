import numpy as np
import matplotlib.pyplot as plt

# 定义函数
def f(x):
    return 0.5 + ((x / 200 - 1) ** 3) * 0.5

# 创建 x 的值范围
x = np.arange(1, 201)  # x 从 1 到 200

# 计算对应的 y 值
y = f(x)

# 绘制图像
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=r'$y=0.5+\left(\frac{x}{200}-1\right)^{3} \times 0.5$', color='blue', linewidth=2)

# 添加标题和标签
plt.title('Plot of the Function', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)

# 添加图例
plt.legend()

# 显示网格
plt.grid(True)

# 显示图像
plt.show()