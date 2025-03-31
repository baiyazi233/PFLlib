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

import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientCFL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        # 正则化参数，用于模型的更新
        self.mu = args.mu
        self.lambda_ = args.lambda_
        # 聚类模型
        self.cluster_model = None
        # 全局模型
        self.global_model = None

    def train(self):
        trainloader = self.load_train_data()
        # cluster_model和global_model不需要训练，保持eval模式
        self.cluster_model.eval()
        self.global_model.eval()
        
        start_time = time.time()
        for epoch in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                # 计算与聚类模型的参数差异（正则项）
                cluster_loss = 0
                for p1, p2 in zip(self.model.parameters(), self.cluster_model.parameters()):
                    cluster_loss += 0.5 * self.mu * torch.sum((p1 - p2) ** 2)
                
                # 计算与全局模型的参数差异（正则项）
                global_loss = 0
                for p1, p2 in zip(self.model.parameters(), self.global_model.parameters()):
                    global_loss += 0.5 * self.lambda_ * torch.sum((p1 - p2) ** 2)
                
                # 总损失 = 分类损失 + 聚类正则项 + 全局正则项
                total_loss = loss + cluster_loss + global_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_cluster_model(self, cluster_model):
        # 设置聚类模型
        self.cluster_model = cluster_model
        # 设置聚类模型参数
        self.set_parameters(self.cluster_model)

    def set_global_model(self, global_model):
        # 设置全局模型
        self.global_model = global_model
        # 全局模型用于本地模型正则项更新，不需要设置参数