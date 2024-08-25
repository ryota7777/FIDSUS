import torch
import time
import copy
import random
import numpy as np
from torch.utils.data import DataLoader

from flcore.clients.clientFIDSUS import clientFIDSUS
from flcore.servers.serverbase import Server

import torch.nn as nn

class FIDSUS(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_clients(clientFIDSUS)
        self.P = torch.diag(torch.ones(self.num_clients, device=self.device))
        self.uploaded_ids = []
        self.M = min(args.M, self.num_join_clients)
        self.client_models = [copy.deepcopy(self.global_model) for _ in range(self.num_clients)]
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []
        self.CEloss = nn.CrossEntropyLoss()
        self.server_learning_rate = args.server_learning_rate
        self.head = self.client_models[0].head
        self.opt_h = torch.optim.SGD(self.head.parameters(), lr=self.server_learning_rate)


    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate_personalized()
            for client in self.selected_clients:
                client.train()
            self.receive_models()
            self.aggregate_parameters()
            self.train_head()
            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        self.save_results()

    def send_models(self):
        assert (len(self.selected_clients) > 0)
        for client in self.clients:
            start_time = time.time()

            M_ = min(self.M, len(self.uploaded_ids))
            indices = torch.topk(self.P[client.id], M_).indices.tolist()
            send_ids = []
            send_models = []
            for i in indices:
                send_ids.append(i)
                send_models.append(self.client_models[i])

            client.receive_models(send_ids, send_models)
            client.set_parameters(self.head)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int(self.client_activity_rate * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_protos = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                for cc in client.protos.keys():
                    y = torch.tensor(cc, dtype=torch.int64, device=self.device)
                    self.uploaded_protos.append((client.protos[cc], y))
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
                self.client_models[client.id] = copy.deepcopy(client.model)
                self.P[client.id] += client.weight_vector
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples


    def train_head(self):
        proto_loader = DataLoader(self.uploaded_protos, self.batch_size, drop_last=False, shuffle=True)
        for p, y in proto_loader:
            out = self.head(p)
            loss = self.CEloss(out, y)
            self.opt_h.zero_grad()
            loss.backward()
            self.opt_h.step()
    def train_metrics_personalized(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics_personalized()
            num_samples.append(ns)
            losses.append(cl * 1.0)
        ids = [c.id for c in self.clients]
        return ids, num_samples, losses
    def test_metrics_personalized(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics_personalized()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        return ids, num_samples, tot_correct, tot_auc

    def evaluate_personalized(self, acc=None, loss=None):
        stats = self.test_metrics_personalized()
        stats_train = self.train_metrics_personalized()
        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)
        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

