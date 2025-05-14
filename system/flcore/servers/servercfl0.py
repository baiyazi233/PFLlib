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
import numpy as np
import torch
from flcore.servers.serverbase import Server
from sklearn.cluster import AgglomerativeClustering
from threading import Thread
from flcore.clients.clientcfl import clientCFL
import copy

class FedCFL0(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # 初始化聚类相关参数
        self.cluster_models = None
        self.cluster_labels = None
        self.cosine_threshold = args.cosine_threshold  # 聚类相似度阈值
        self.min_cluster_size = args.min_cluster_size  # 最小聚类大小
        self.cluster_interval = getattr(args, 'cluster_interval', 1)  # 聚类间隔，默认1
        
        # 选择参与训练的客户端
        self.set_slow_clients()
        self.set_clients(clientCFL)
        
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        
        self.Budget = []
        self.initialize_cluster_models()

    def initialize_cluster_models(self):
        """初始化聚类模型，初始时将所有客户端视为一个簇"""
        self.cluster_models = [copy.deepcopy(self.global_model)]
        self.cluster_labels = [0] * self.num_clients

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_cluster_models()
            
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
            
            for client in self.selected_clients:
                client.train()
            
            # 接收客户端更新
            self.receive_client_updates()
            
            # 聚类客户端
            if i % self.cluster_interval == 0:
                self.cluster_clients()
            
            # 聚合每个簇的模型
            self.aggregate_cluster_models()
            
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])
            
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        
        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        
        self.save_results()
        self.save_cluster_models()

    def send_cluster_models(self):
        """向每个客户端发送其所属簇的模型"""
        for client in self.selected_clients:
            cluster_id = self.cluster_labels[client.id]
            
            # 如果cluster_id超出了cluster_models的范围，使用全局模型
            if cluster_id >= len(self.cluster_models):
                client.set_cluster_model(copy.deepcopy(self.global_model))
            else:
                client.set_cluster_model(copy.deepcopy(self.cluster_models[cluster_id]))

    def receive_client_updates(self):
        """收集客户端的模型更新和参数"""
        self.client_updates = []
        self.client_params = []
        
        for client in self.selected_clients:
            self.client_updates.append(client.get_update_direction())
            self.client_params.append(client.get_parameters())

    def cluster_clients(self):
        """基于客户端更新方向进行聚类"""
        # 计算客户端更新之间的余弦相似度
        similarities = self.compute_pairwise_similarities()
        
        # 基于相似度进行层次聚类
        clustering = AgglomerativeClustering(
            n_clusters=None, 
            metric='precomputed', 
            linkage='complete',
            distance_threshold=1.0 - self.cosine_threshold  # 转换为距离度量
        )
        
        # 执行聚类
        client_indices = [client.id for client in self.selected_clients]
        labels = clustering.fit_predict(1.0 - similarities)  # 转换为距离矩阵
        
        # 更新客户端聚类标签
        for i, client_id in enumerate(client_indices):
            self.cluster_labels[client_id] = labels[i]
        
        # 检查并分裂不匹配的簇
        self.check_and_split_clusters()

    def compute_pairwise_similarities(self):
        """计算客户端更新之间的余弦相似度矩阵"""
        n_clients = len(self.client_updates)
        similarities = np.zeros((n_clients, n_clients))
        
        for i in range(n_clients):
            for j in range(i, n_clients):
                sim = self.cosine_similarity(self.client_updates[i], self.client_updates[j])
                similarities[i, j] = sim
                similarities[j, i] = sim
        
        return similarities

    def cosine_similarity(self, update1, update2):
        """计算两个模型更新之间的余弦相似度"""
        dot_product = 0.0
        norm1 = 0.0
        norm2 = 0.0
        
        for param1, param2 in zip(update1, update2):
            dot_product += torch.sum(param1 * param2).item()
            norm1 += torch.sum(param1 ** 2).item()
            norm2 += torch.sum(param2 ** 2).item()
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (np.sqrt(norm1) * np.sqrt(norm2))

    def aggregate_cluster_parameters(self, cluster_params):
        """聚合簇内客户端的模型参数"""
        for param in cluster_params[0]:
            param.data = torch.zeros_like(param.data)
            
        for params in cluster_params:
            for param, aggregated_param in zip(params, cluster_params[0]):
                aggregated_param.data += param.data
                
        for param in cluster_params[0]:
            param.data /= len(cluster_params)
            
        return cluster_params[0]

    def aggregate_cluster_models(self):
        """为每个簇聚合模型参数"""
        cluster_indices = {}
        for client in self.selected_clients:
            cluster_id = self.cluster_labels[client.id]
            if cluster_id not in cluster_indices:
                cluster_indices[cluster_id] = []
            cluster_indices[cluster_id].append(client.id)
        
        # 为每个簇聚合模型
        for cluster_id, client_ids in cluster_indices.items():
            if len(client_ids) >= self.min_cluster_size:
                # 获取该簇中所有客户端的参数
                cluster_params = [self.client_params[i] for i, client in enumerate(self.selected_clients) 
                                 if client.id in client_ids]
                
                # 聚合参数
                aggregated_params = self.aggregate_cluster_parameters(cluster_params)
                
                # 更新簇模型
                if cluster_id < len(self.cluster_models):
                    self.cluster_models[cluster_id] = aggregated_params
                else:
                    self.cluster_models.append(aggregated_params)

    def check_and_split_clusters(self):
        """检查并分裂不匹配的簇"""
        # 统计每个簇的客户端数量
        cluster_counts = {}
        for client in self.selected_clients:
            cluster_id = self.cluster_labels[client.id]
            if cluster_id not in cluster_counts:
                cluster_counts[cluster_id] = 0
            cluster_counts[cluster_id] += 1
        
        # 检查每个簇的大小，如果小于min_cluster_size则分裂
        for cluster_id, count in cluster_counts.items():
            if count < self.min_cluster_size:
                # 将该簇的客户端重新分配到其他簇
                for client in self.selected_clients:
                    if self.cluster_labels[client.id] == cluster_id:
                        # 这里可以随机分配或根据相似度分配到其他簇
                        self.cluster_labels[client.id] = max(cluster_counts.keys()) + 1

    def evaluate_clusters(self):
        """评估每个簇的模型性能"""
        for cluster_id, model in enumerate(self.cluster_models):
            # 设置客户端使用该簇的模型进行评估
            for client in self.clients:
                if self.cluster_labels[client.id] == cluster_id:
                    client.set_parameters(copy.deepcopy(model))
            
            # 评估该簇的性能
            acc, loss = self.evaluate(selected=self.clients)
            print(f"Cluster {cluster_id} - Test accuracy: {acc:.4f}, Test loss: {loss:.4f}")