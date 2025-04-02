import h5py
import matplotlib.pyplot as plt
import numpy as np

# 加载第一个文件的数据
with h5py.File('Cifar10_FedAvg_test_LeNet_200r_50d_0.005lr.h5', 'r') as f1:
    test_acc1 = f1['rs_test_acc'][:]
    test_auc1 = f1['rs_test_auc'][:]
    train_loss1 = f1['rs_train_loss'][:]

# 加载第二个文件的数据
with h5py.File('Cifar10_FedCFL_test_LeNet_200r_50d_0.005lr.h5', 'r') as f2:
    test_acc2 = f2['rs_test_acc'][:]
    test_auc2 = f2['rs_test_auc'][:]
    train_loss2 = f2['rs_train_loss'][:]

# 确保数据长度一致（假设两个文件训练轮次相同）
epochs = range(1, len(test_acc1) + 1)
plt.figure(figsize=(10, 5))

# Test Accuracy 对比
plt.plot(epochs, test_acc1, 'b-', label='Test Acc (File 1)')
plt.plot(epochs, test_acc2, 'b--', label='Test Acc (File 2)')

# Test AUC 对比
# plt.plot(epochs, test_auc1, 'g-', label='Test AUC (File 1)')
# plt.plot(epochs, test_auc2, 'g--', label='Test AUC (File 2)')

# Train Loss 对比
plt.plot(epochs, train_loss1, 'r-', label='Train Loss (File 1)')
plt.plot(epochs, train_loss2, 'r--', label='Train Loss (File 2)')

plt.title('Training Metrics Comparison', fontsize=12)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Value', fontsize=10)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在右侧
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Test Accuracy
ax1.plot(epochs, test_acc1, 'b-', label='FedAvg')
ax1.plot(epochs, test_acc2, 'b--', label='FedCFL')
ax1.set_ylabel('Test Accuracy')
ax1.legend()

# Test AUC
# ax2.plot(epochs, test_auc1, 'g-', label='File 1')
# ax2.plot(epochs, test_auc2, 'g--', label='File 2')
# ax2.set_ylabel('Test AUC')
# ax2.legend()

# Train Loss
ax2.plot(epochs, train_loss1, 'r-', label='FedAvg')
ax2.plot(epochs, train_loss2, 'r--', label='FedCFL')
ax2.set_ylabel('Train Loss')
ax2.set_xlabel('Epochs')
ax2.legend()

plt.suptitle('Metrics Comparison Between Two Models', y=1.02)
plt.tight_layout()
plt.show()