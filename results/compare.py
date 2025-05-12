import h5py
import matplotlib.pyplot as plt
import numpy as np

# 加载第一个文件的数据
with h5py.File('MNIST_FedAvg_test_LeNet_200r_50d.h5', 'r') as f1:
    test_acc1 = f1['rs_test_acc'][:]
    test_auc1 = f1['rs_test_auc'][:]
    train_loss1 = f1['rs_train_loss'][:]

# 加载第二个文件的数据
with h5py.File('MNIST_FedCFL_test_LeNet_200r_50d.h5', 'r') as f2:
    test_acc2 = f2['rs_test_acc'][:]
    test_auc2 = f2['rs_test_auc'][:]
    train_loss2 = f2['rs_train_loss'][:]

# 加载第三个文件的数据
with h5py.File('MNIST_FedProx_test_LeNet_200r_50d.h5', 'r') as f3:
    test_acc3 = f3['rs_test_acc'][:]
    test_auc3 = f3['rs_test_auc'][:]
    train_loss3 = f3['rs_train_loss'][:]

# 加载第四个文件的数据
with h5py.File('MNIST_FedRep_test_LeNet_200r_50d.h5', 'r') as f4:
    test_acc4 = f4['rs_test_acc'][:]
    test_auc4 = f4['rs_test_auc'][:]
    train_loss4 = f4['rs_train_loss'][:]

# 加载第五个文件的数据
with h5py.File('MNIST_PerAvg_test_LeNet_200r_50d.h5', 'r') as f5:
    test_acc5 = f5['rs_test_acc'][:]
    test_auc5 = f5['rs_test_auc'][:]
    train_loss5 = f5['rs_train_loss'][:]

# 确保数据长度一致（假设所有文件训练轮次相同）
epochs = range(1, len(test_acc1) + 1)

# 创建两个子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Test Accuracy
ax1.plot(epochs, test_acc1, '-', label='FedAvg', color='#1f77b4')  # 蓝色
ax1.plot(epochs, test_acc2, '--', label='FedCFL', color='#ff7f0e')  # 橙色
ax1.plot(epochs, test_acc3, ':', label='FedProx', color='#2ca02c')  # 绿色
ax1.plot(epochs, test_acc4, '-.', label='FedRep', color='#d62728')  # 红色
ax1.plot(epochs, test_acc5, '--', label='PerAvg', color='#9467bd')  # 紫色
ax1.set_ylabel('Test Accuracy')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# Train Loss
ax2.plot(epochs, train_loss1, '-', label='FedAvg', color='#1f77b4')
ax2.plot(epochs, train_loss2, '--', label='FedCFL', color='#ff7f0e')
ax2.plot(epochs, train_loss3, ':', label='FedProx', color='#2ca02c')
ax2.plot(epochs, train_loss4, '-.', label='FedRep', color='#d62728')
ax2.plot(epochs, train_loss5, '--', label='PerAvg', color='#9467bd')
ax2.set_ylabel('Train Loss')
ax2.set_xlabel('Epochs')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

plt.suptitle('Performance Comparison of Different Federated Learning Methods', y=1.02)
plt.tight_layout()
plt.show()

# 保存图片
# plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')