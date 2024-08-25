
import copy
import torch
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client


class clientMOON(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.tau = args.tau
        self.mu = args.mu

        self.global_model = None
        self.old_model = copy.deepcopy(self.model)

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)#新的特征表示
                output = self.model.head(rep)
                loss = self.loss(output, y)#计算新的loss

                rep_old = self.old_model.base(x).detach()#老的特征表示
                rep_global = self.global_model.base(x).detach()#全局特征表示
                loss_con = - torch.log(torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) / (torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) + torch.exp(F.cosine_similarity(rep, rep_old) / self.tau)))
                #loss_con = [e ^ ((rep与rep_global的余弦相似度)/tau)]/ {[e ^ ((rep与rep_global的余弦相似度)/tau)] + [e ^ ((rep与rep_old的余弦相似度)/tau)]}
                #loss_con作为正则项调整loss
                loss += self.mu * torch.mean(loss_con)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()
        self.old_model = copy.deepcopy(self.model)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

        self.global_model = model

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)#得到特征表示
                output = self.model.head(rep)
                loss = self.loss(output, y)#计算损失值

                rep_old = self.old_model.base(x).detach()#
                rep_global = self.global_model.base(x).detach()
                loss_con = - torch.log(torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) / (torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) + torch.exp(F.cosine_similarity(rep, rep_old) / self.tau)))
                loss += self.mu * torch.mean(loss_con)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num