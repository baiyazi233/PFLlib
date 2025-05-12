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
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        print("train starting", flush=True)
        for i in range(self.global_rounds+1):
            print(f"train loop i={i}", flush=True)
            s_t = time.time()
            self.selected_clients = self.select_clients()
            # print("selected_clients done", flush=True)
            self.send_models()
            print("send_models done", flush=True)

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------", flush=True)
                print("\nEvaluate global model", flush=True)
                self.evaluate()
                print("evaluate done", flush=True)

            for client in self.selected_clients:
                # print(f"client {client.id} train start", flush=True)
                client.train()
                # print(f"client {client.id} train end", flush=True)

            self.receive_models()
            print("receive_models done", flush=True)
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
                print("call_dlg done", flush=True)
            self.aggregate_parameters()
            print("aggregate_parameters done", flush=True)

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1], flush=True)

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                print("auto_break triggered", flush=True)
                break

        print("\nBest accuracy.", flush=True)
        print(max(self.rs_test_acc), flush=True)
        print("\nAverage time cost per round.", flush=True)
        print(sum(self.Budget[1:])/len(self.Budget[1:]), flush=True)

        self.save_results()
        print("save_results done", flush=True)
        self.save_global_model()
        print("save_global_model done", flush=True)

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------", flush=True)
            print("\nEvaluate new clients", flush=True)
            self.evaluate()
            print("evaluate new clients done", flush=True)
