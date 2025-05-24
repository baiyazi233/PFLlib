import h5py
import matplotlib.pyplot as plt
import numpy as np

# 加载第一个文件的数据
with h5py.File('MNIST_FedADC_test_LeNet_200r_50d.h5', 'r') as f1:
    test_acc1 = f1['rs_test_acc'][:]

# 加载第二个文件的数据
with h5py.File('MNIST_FedAC_test_LeNet_200r_50d.h5', 'r') as f2:
    test_acc2 = f2['rs_test_acc'][:]

# 加载第三个文件的数据
with h5py.File('MNIST_FedAvg_test_LeNet_200r_50d.h5', 'r') as f3:
    test_acc3 = f3['rs_test_acc'][:]


# # # 加载第一个文件的数据
# with h5py.File('Cifar10_FedADC_test_LeNet_200r_50d.h5', 'r') as f1:
#     test_acc1 = f1['rs_test_acc'][:]

# # 加载第二个文件的数据
# with h5py.File('Cifar10_FedAC_test_LeNet_200r_50d.h5', 'r') as f2:
#     test_acc2 = f2['rs_test_acc'][:]

# # 加载第三个文件的数据
# with h5py.File('Cifar10_FedAvg_test_LeNet_200r_50d.h5', 'r') as f3:
#     test_acc3 = f3['rs_test_acc'][:]

# 确保数据长度一致（假设所有文件训练轮次相同）
epochs = range(1, len(test_acc1) + 1)

# 创建图表
plt.figure(figsize=(12, 6))

# Test Accuracy
plt.plot(epochs, test_acc1, '-', label='FedADC', color='#1f77b4')  # 蓝色
plt.plot(epochs, test_acc2, '--', label='FedAC', color='#d62728')  # 红色
plt.plot(epochs, test_acc3, ':', label='FedAvg', color='#2ca02c')  # 绿色


plt.ylabel('Test Accuracy')
plt.xlabel('Epochs')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.title('Performance Comparison of Different Federated Learning Methods on MNIST with alpha = 0.1')
# plt.title('Performance Comparison of Different Federated Learning Methods on MNIST with n = 3')
# plt.title('Performance Comparison of Different Federated Learning Methods on CIFAR-10 with alpha = 0.1')
# plt.title('Performance Comparison of Different Federated Learning Methods on CIFAR-10 with n = 3')

plt.tight_layout()
plt.show()

# 打印每种方法的最大准确率
print("\nMaximum Test Accuracy for each method:")
print(f"FedADC: {max(test_acc1):.4f}")
print(f"FedAC: {max(test_acc2):.4f}")


# 保存图片
# plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')