import h5py
import matplotlib.pyplot as plt
import numpy as np

# 加载第一个文件的数据
with h5py.File('Cifar10_FedAvg_test_LeNet_200r_50d_4n.h5', 'r') as f1:
    test_acc1 = f1['rs_test_acc'][:]

# 加载第二个文件的数据
with h5py.File('Cifar10_FedCFL_test_LeNet_200r_50d_4n.h5', 'r') as f2:
    test_acc2 = f2['rs_test_acc'][:]

# 加载第三个文件的数据
with h5py.File('Cifar10_FedProx_test_LeNet_200r_50d_4n.h5', 'r') as f3:
    test_acc3 = f3['rs_test_acc'][:]

# 加载第四个文件的数据
with h5py.File('Cifar10_PerAvg_test_LeNet_200r_50d_4n.h5', 'r') as f4:
    test_acc4 = f4['rs_test_acc'][:]


# 确保数据长度一致（假设所有文件训练轮次相同）
epochs = range(1, len(test_acc1) + 1)

# 创建图表
plt.figure(figsize=(12, 6))

# Test Accuracy
plt.plot(epochs, test_acc1, '-', label='FedAvg', color='#1f77b4')  # 蓝色
plt.plot(epochs, test_acc2, '--', label='FedAC', color='#d62728')  # 红色
plt.plot(epochs, test_acc3, ':', label='FedProx', color='#2ca02c')  # 绿色
plt.plot(epochs, test_acc4, '-.', label='CFL', color='#9467bd')  # 紫色


plt.ylabel('Test Accuracy')
plt.xlabel('Epochs')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.title('Performance Comparison of Different Federated Learning Methods on CIFAR-10 with n = 4')
plt.tight_layout()
plt.show()

# 打印每种方法的最大准确率
print("\nMaximum Test Accuracy for each method:")
print(f"FedAvg: {max(test_acc1):.4f}")
print(f"FedAC: {max(test_acc2):.4f}")
print(f"FedProx: {max(test_acc3):.4f}")
print(f"CFL: {max(test_acc4):.4f}")


# 保存图片
# plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')