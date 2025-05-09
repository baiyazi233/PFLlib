import time
import torch
import torch.nn.functional as F
from flcore.clients.clientadmm import ClientADMM
from flcore.servers.serverbase import Server
from threading import Thread

class FedADMM(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # 参数设置
        self.gamma = 0.1  # γ > 0
        self.rho = 0.9    # ρ ∈ (0,1)
        self.current_round = 0  # 当前轮次

        # 选择慢速客户端
        self.set_slow_clients()
        
        # 设置客户端，传入总样本数
        self.set_clientsADMM(ClientADMM)

        # 存储客户端的本地模型和拉格朗日乘子
        self.client_models = []
        self.client_lambdas = []
        self.selected_clients_h = []  # 存储被选中且Hk>0的客户端

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def calculate_threshold(self):
        """计算第t轮的阈值ξ^t = γρ^t"""
        return self.gamma * (self.rho ** self.current_round)

    def calculate_hk(self, client, x, y):
        """计算客户端的Hk值
        Hk = ||L(θk,Φ) - (1/K)∑L(θk',Φ)||² - ξ^t
        其中L是损失函数
        """
        # 计算当前客户端的损失 L(θk,Φ)
        client_output = client.model(x)
        client_loss = F.cross_entropy(client_output, y)
        
        # 计算所有客户端的平均损失 (1/K)∑L(θk',Φ)
        all_losses = []
        for c in self.selected_clients:
            with torch.no_grad():
                output = c.model(x)
                loss = F.cross_entropy(output, y)
                all_losses.append(loss)
        avg_loss = torch.stack(all_losses).mean()
        
        # 计算Hk，使用当前轮次的阈值
        diff = client_loss - avg_loss
        threshold = self.calculate_threshold()
        hk = (diff ** 2) - threshold
        
        print(f"Client {client.id} - Loss: {client_loss:.4f}, Avg Loss: {avg_loss:.4f}, Hk: {hk.item():.4f}, Threshold ξ^{self.current_round}: {threshold:.4f}")
        return hk.item()

    def train(self):
        for i in range(self.global_rounds+1):
            self.current_round = i  # 更新当前轮次
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            # 客户端训练
            for client in self.selected_clients:
                client.train()

            # 清空上一轮的数据
            self.client_models = []
            self.client_lambdas = []
            self.selected_clients_h = []
            
            # 计算每个客户端的Hk并筛选
            if len(self.selected_clients) > 0:
                # 获取一个batch的数据用于计算Hk
                sample_client = self.selected_clients[0]
                trainloader = sample_client.load_train_data()
                x, y = next(iter(trainloader))
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                    x = x[0]
                x = x.to(self.device)
                y = y.to(self.device)

                # 根据Hk筛选客户端
                threshold = self.calculate_threshold()
                print(f"\nRound {i} - Current threshold ξ^t: {threshold:.4f}")
                
                for client in self.selected_clients:
                    hk = self.calculate_hk(client, x, y)
                    if hk > 0:
                        self.selected_clients_h.append(client)
                        
                print(f"Selected {len(self.selected_clients_h)}/{len(self.selected_clients)} clients based on Hk")
            
            # 接收客户端模型
            self.receive_models()
            
            # 聚合参数
            if len(self.selected_clients_h) > 0:
                self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

    def aggregate_parameters(self):
        """根据公式(5)聚合参数：
        Θ^(i+1) = (1/Σβ_n) * Σ(β_n * θ_n^(i+1) + λ_n^i)
        """
        assert len(self.client_models) > 0, "No client models received"
        
        beta_sum = sum(client.beta for client in self.selected_clients_h)
        
        # 初始化聚合后的参数字典
        aggregated_parameters = {}
        for name, param in self.global_model.state_dict().items():
            if torch.is_floating_point(param):  # 只处理浮点类型参数
                aggregated_parameters[name] = torch.zeros_like(param)
        
        # 根据公式(5)聚合参数
        for client_idx, client in enumerate(self.selected_clients_h):
            client_model = self.client_models[client_idx]
            client_lambda = self.client_lambdas[client_idx]
            
            for name, param in client_model.items():
                if name in aggregated_parameters:
                    # β_n * θ_n^(i+1) + λ_n^i
                    weighted_param = client.beta * param + client_lambda[name]
                    aggregated_parameters[name] += weighted_param
        
        # 除以β之和
        for name in aggregated_parameters:
            aggregated_parameters[name] /= beta_sum
        
        # 更新全局模型参数
        self.global_model.load_state_dict(aggregated_parameters)

    def send_models(self):
        """向选中的客户端发送全局模型"""
        for client in self.selected_clients:
            client.set_parameters(self.global_model)

    def receive_models(self):
        """接收Hk>0的客户端的本地模型和拉格朗日乘子"""
        for client in self.selected_clients_h:
            # 获取客户端的本地模型参数
            local_model = {}
            for name, param in client.model.state_dict().items():
                if torch.is_floating_point(param):
                    local_model[name] = param.clone()
            self.client_models.append(local_model)
            
            # 获取客户端的拉格朗日乘子
            local_lambda = {}
            for name, lambda_param in client.lambda_k.items():
                local_lambda[name] = lambda_param.clone()
            self.client_lambdas.append(local_lambda) 