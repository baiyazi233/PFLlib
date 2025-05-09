import copy
import time
import torch
import torch.nn as nn
from flcore.clients.clientbase import Client

class ClientADMM(Client):
    def __init__(self, args, id, train_samples, test_samples, total_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # ADMM 参数
        self.beta = 0.1  # 二次惩罚项系数 beta_k
        self.total_samples = total_samples  # 所有客户端的总样本数
        
        # 计算alpha_k = |D_k| / sum(|D_k'|)
        self.alpha = self.train_samples / self.total_samples
        self.max_iter = 10  # ADMM最大迭代次数
        
        # 初始化变量
        self.theta = {}  # 模型参数 theta_k
        self.theta_old = {}  # 上一轮的参数 Theta^i
        self.lambda_k = {}  # 拉格朗日乘子 lambda_k^i
        self.global_model = {}  # 服务器下发的全局模型参数 Theta^(i+1)
        
        # 为每个参数初始化变量
        for name, param in self.model.state_dict().items():
            if torch.is_floating_point(param):  # 只处理浮点类型参数
                self.theta[name] = param.data.clone()
                self.theta_old[name] = param.data.clone()
                self.lambda_k[name] = torch.zeros_like(param.data)
                self.global_model[name] = param.data.clone()

    def train(self):
        trainloader = self.load_train_data()
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 主循环
        for epoch in range(self.local_epochs):
            # 保存上一轮的参数
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.theta_old[name] = param.data.clone()
            
            for batch_idx, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                # 更新参数
                self.update_parameters(x, y)
        
        # 计算训练时间
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def update_parameters(self, x, y):
        self.model.zero_grad()
        
        # 计算损失函数 f_k(theta_k)
        output = self.model(x)
        loss = self.loss(output, y)
        
        # 计算完整的目标函数
        total_loss = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # alpha_k * f_k(theta_k)
                total_loss += self.alpha * loss
                
                # <lambda_k^i, theta_k - Theta^i>
                lambda_term = torch.sum(self.lambda_k[name] * (param - self.theta_old[name]))
                total_loss += lambda_term
                
                # (beta_k/2) * ||theta_k - Theta^i||^2
                quad_term = (self.beta/2) * torch.norm(param - self.theta_old[name])**2
                total_loss += quad_term
        
        # 反向传播
        total_loss.backward()
        
        # 更新参数 theta_k
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grad = param.grad.data
                param.data = param.data - self.learning_rate * grad
                self.theta[name] = param.data.clone()

    def set_parameters(self, model):
        """从服务器接收全局模型参数并更新lambda"""
        # 首先保存全局模型参数
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
            
        # 更新global_model字典
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.global_model[name] = param.data.clone()
        
        # 更新lambda根据公式(6)
        self.update_lambda()

    def update_lambda(self):
        """根据公式(6)更新对偶变量lambda"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # λ_k^(i+1) = λ_k^i + β_k(θ_k^(i+1) - Θ_k^(i+1))
                self.lambda_k[name] = self.lambda_k[name] + self.beta * (self.theta[name] - self.global_model[name]) 