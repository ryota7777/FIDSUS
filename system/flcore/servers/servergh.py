
import time
import torch
import torch.nn as nn
from flcore.clients.clientgh import clientGH
from flcore.servers.serverbase import Server
from torch.utils.data import DataLoader


class FedGH(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.global_model = None

        self.set_clients(clientGH)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.CEloss = nn.CrossEntropyLoss()
        self.server_learning_rate = args.server_learning_rate

        self.head = self.clients[0].model.head
        self.opt_h = torch.optim.SGD(self.head.parameters(), lr=self.server_learning_rate)


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()
                client.collect_protos()



            self.receive_protos()
            self.train_head()
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])


        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.head)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):#得到类别特征表示与类别标签
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            for cc in client.protos.keys():#cc：样本标签类别
                y = torch.tensor(cc, dtype=torch.int64, device=self.device) #y：tensor张量的标签
                self.uploaded_protos.append((client.protos[cc], y))# 将特征表示与对应的类别标签写入字典
            
    def train_head(self):
        proto_loader = DataLoader(self.uploaded_protos, self.batch_size, drop_last=False, shuffle=True)
        #加载前面所提取的特征表示
        for p, y in proto_loader:
            out = self.head(p)#前向传播预测
            loss = self.CEloss(out, y)#计算loss
            self.opt_h.zero_grad()#清空head梯度
            loss.backward()#反向传播计算梯度
            self.opt_h.step()#更新参数
