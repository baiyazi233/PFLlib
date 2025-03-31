# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import time
from flcore.clients.clientcfl import clientCFL
from flcore.servers.serverbase import Server
from threading import Thread
from torch.nn.utils import parameters_to_vector
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import copy
import torch
import random
import numpy as np

class FedCFL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.num_clusters = args.num_clusters
        self.R = self.generate_R_matrix()
        self.cluster_models = self.generate_cluster_models()
        # 随机采样Sd个局部模型的参数，将其展平拼接，得到Wd，维度为|Sd|×dim(ω)
        self.Wd = self.generate_Wd()
        # 使用PCA对Wd进行降维
        self.M = self.generate_M()
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientCFL)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_cluster_and_global_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            # 更新分配矩阵R
            self.update_R()
            # 更新聚合模型
            self.update_cluster_models()
            # 更新全局模型
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    # 生成 R 矩阵
    def generate_R_matrix(self):
        # 将所有客户端随机分配到 K 个聚类中
        R = np.zeros((self.num_clients, self.num_clusters), dtype=int)
        for i in range(self.num_clients):
            # 随机选择一个聚类
            k = np.random.randint(0, self.num_clusters)
            R[i, k] = 1
        return R
    
    # 生成聚类模型
    def generate_cluster_models(self):
        # 对全局模型进行 K 次随机 Dropout 操作，生成 K 个不同的局部模型
        cluster_models = []
        for i in range(self.num_clusters):
            cluster_model = copy.deepcopy(self.global_model)
            # 将模型设置为训练模式以启用dropout
            cluster_model.train()
            # 对模型进行前向传播以应用dropout
            with torch.no_grad():
                # 创建一个随机输入来触发dropout，MNIST的输入维度是1x28x28
                dummy_input = torch.randn(1, 1, 28, 28).to(cluster_model.parameters().__next__().device)
                _ = cluster_model(dummy_input)
            cluster_model.eval()  # 将模型设置回评估模式
            cluster_models.append(cluster_model)
        return cluster_models

    # 发送聚类模型和全局模型
    def send_cluster_and_global_models(self):
        assert (len(self.clients) > 0)
        for client in self.clients:
          start_time = time.time()
          # 根据 R 矩阵确定客户端所属的聚类
          cluster_id = np.argmax(self.R[client.id])
          client.set_cluster_model(self.cluster_models[cluster_id])
          client.set_global_model(self.global_model)
          client.send_time_cost['num_rounds'] += 1
          client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    # 随机采样Sd个局部模型的参数，将其展平拼接，得到Wd，维度为|Sd|×dim(ω)
    def generate_Wd(self):
        Wd = []
        sampled_models = random.sample(self.cluster_models, int(self.num_clients/2))  # 将浮点数转换为整数
        for model in sampled_models:
            flattened_params = parameters_to_vector(model.parameters())
            Wd.append(flattened_params)
        Wd = torch.stack(Wd)
        return Wd

    # 使用PCA对Wd进行降维
    def generate_M(self):
        scaler = StandardScaler()
        Wd_normalized = scaler.fit_transform(self.Wd.detach().cpu().numpy())  # 先分离梯度，再转换为 NumPy 并标准化
        # 使用PCA对Wd进行降维
        D = int(self.num_clients/2)  # 将浮点数转换为整数
        pca = PCA(n_components=D)
        pca.fit(Wd_normalized)
        return pca.components_

    # 更新分配矩阵R
    def update_R(self):
        # 计算每个客户端与每个聚类中心的距离
        distances = []
        for i in range(self.num_clients):
            for k in range(self.num_clusters):
                distances.append(self.compute_similarity(self.clients[i].model, self.cluster_models[k]))
        distances = torch.tensor(distances).reshape(self.num_clients, self.num_clusters)
        # 更新分配矩阵R
        self.R = torch.zeros((self.num_clients, self.num_clusters), dtype=int)
        for i in range(self.num_clients):
            self.R[i, torch.argmax(distances[i])] = 1


    # 计算模型相似度
    def compute_similarity(self, model1, model2):
        # 计算两个模型的参数向量之间的余弦相似度，使用PCA降维后的矩阵M
        params1 = parameters_to_vector(model1.parameters()).detach()
        params2 = parameters_to_vector(model2.parameters()).detach()
        # 将M转换为PyTorch张量
        M_tensor = torch.from_numpy(self.M).float().to(params1.device)
        # 计算降维后的参数向量
        reduced_params1 = torch.matmul(M_tensor, params1)
        reduced_params2 = torch.matmul(M_tensor, params2)
        # 计算余弦相似度
        return torch.nn.functional.cosine_similarity(reduced_params1.unsqueeze(0), reduced_params2.unsqueeze(0))

    # 更新聚合模型
    def update_cluster_models(self):
        for k in range(self.num_clusters):
            # 计算每个聚类中所有客户端的模型参数的平均值
            cluster_models = [self.clients[i].model for i in range(self.num_clients) if self.R[i, k] == 1]
            # 计算平均模型
            avg_model = self.average_cluster_models(cluster_models)
            if avg_model is not None:  # 只有当簇不为空时才更新
                self.cluster_models[k] = avg_model

    # 计算cluster_models列表中模型参数的平均值
    def average_cluster_models(self, cluster_models):
        if not cluster_models:  # 如果簇为空，返回None
            return None
        # 使用簇中的第一个模型作为基础模型
        avg_model = copy.deepcopy(cluster_models[0])
        for param_idx, param in enumerate(avg_model.parameters()):
            # 收集所有模型中对应参数
            param_list = []
            for model in cluster_models:
                param_list.append(list(model.parameters())[param_idx])
            # 计算平均值
            param.data = torch.stack(param_list).mean(dim=0)
        return avg_model
