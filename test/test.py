import torch
import random
from torch.nn.utils import parameters_to_vector
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# 示例：假设有 10 个局部模型，每个模型的参数量为 100
local_models = [torch.nn.Linear(10, 10) for _ in range(10)]  # 假设有 10 个模型
S_d = 3  # 采样 3 个模型
Wd = []
sampled_models = random.sample(local_models, S_d)
for model in sampled_models:
  flattened_params = parameters_to_vector(model.parameters())
  print(flattened_params.shape)
  Wd.append(flattened_params)
Wd = torch.stack(Wd)
print(Wd.shape)  # 输出: torch.Size([3, 110]) (假设每个 Linear(10,10) 有 110 个参数)

scaler = StandardScaler()
Wd_normalized = scaler.fit_transform(Wd.detach().cpu().numpy())  # 转换为 NumPy 并标准化
# 使用PCA对Wd进行降维
D = 2
pca = PCA(n_components=D)
pca.fit(Wd_normalized)
print(pca.components_ @ parameters_to_vector(sampled_models[0].parameters()).detach().cpu().numpy())