import torch
import numpy as np
import time
import copy

from torch import nn

from flcore.clients.clientbase import Client
from torch.utils.data import DataLoader
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
from utils.data_utils import  read_client_data_un
import torch.nn.functional as F
from collections import defaultdict
from sklearn.preprocessing import label_binarize
from sklearn import metrics

class clientFIDSUS(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.num_clients = args.num_clients
        self.old_model = copy.deepcopy(self.model)
        self.received_ids = []
        self.received_models = []
        self.weight_vector = torch.zeros(self.num_clients, device=self.device)
        self.val_ratio = 0.2
        self.train_samples = self.train_samples * (1 - self.val_ratio)
        self.batch_size = args.batch_size
        self.mu = args.mu
        self.model_per = copy.deepcopy(self.model)
        self.optimizer_per = PerturbedGradientDescent(
            self.model_per.parameters(), lr=self.learning_rate, mu=self.mu)
        self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_per,
            gamma=args.learning_rate_decay_gamma
        )
        self.CEloss = nn.CrossEntropyLoss()
        self.head_per = self.model_per.head
        self.opt_h_per = torch.optim.SGD(self.head_per.parameters(), lr=self.learning_rate)

    def train(self):
        trainloader, val_loader = self.load_train_data()
        start_time = time.time()
        self.aggregate_parameters(val_loader)#
        self.clone_model(self.model, self.old_model)
        self.model.train()
        self.model_per.train()
        protos = defaultdict(list)
        protos_per = defaultdict(list)
        max_local_epochs = self.local_epochs
        for epoch in range(max_local_epochs):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                reg = self.model.base(x)
                output = self.model.head(reg)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                reg_per = self.model_per.base(x)
                output = self.model_per.head(reg_per)
                loss = self.loss(output, y)
                self.optimizer_per.zero_grad()
                loss.backward()
                self.optimizer_per.step(self.model_per.parameters(), self.device)
                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(reg[i, :].detach().data)
                    protos_per[y_c].append(reg_per[i, :].detach().data)
        self.protos_g = agg_func(protos)
        self.protos_per = agg_func(protos_per)
        self.protos = aggregation(self.protos_g, self.protos_per)
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def load_train_data(self, batch_size=None):  # 读取client数据集并划分为训练集验证集
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data_un(self.dataset, self.id, is_train=True)
        val_idx = -int(self.val_ratio * len(train_data))
        val_data = train_data[val_idx:]
        train_data = train_data[:val_idx]
        trainloader = DataLoader(train_data, self.batch_size, drop_last=True, shuffle=False)
        val_loader = DataLoader(val_data, self.batch_size, drop_last=self.has_BatchNorm, shuffle=False)
        return trainloader, val_loader
    def receive_models(self, ids, models):
        self.received_ids = ids
        self.received_models = models
    def set_parameters(self, head):
        for new_param, old_param in zip(head.parameters(), self.model.head.parameters()):
            old_param.data = new_param.data.clone()
    def weight_cal(self, val_loader):
        weight_list = []
        L = self.recalculate_loss(self.old_model, val_loader)
        for received_model in self.received_models:
            params_dif = []
            for param_n, param_i in zip(received_model.parameters(), self.old_model.parameters()):
                params_dif.append((param_n - param_i).view(-1))
            params_dif = torch.cat(params_dif)
            weight_list.append(
                (L - self.recalculate_loss(received_model, val_loader)) / (torch.norm(params_dif) + 1e-5))
        self.weight_vector_update(weight_list)
        return torch.tensor(weight_list)

    def weight_vector_update(self, weight_list):
        self.weight_vector = np.zeros(self.num_clients)
        for w, id in zip(weight_list, self.received_ids):
            self.weight_vector[id] += w.item()
        self.weight_vector = torch.tensor(self.weight_vector).to(self.device)
    def recalculate_loss(self, new_model, val_loader):
        L = 0
        for x, y in val_loader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            output = new_model(x)
            loss = self.loss(output, y)
            L += loss.item()
        return L / len(val_loader)

    def add_parameters(self, w, received_model):
        for param, received_param in zip(self.model.parameters(), received_model.parameters()):
            param.data += received_param.data.clone() * w

    def aggregate_parameters(self, val_loader):
        weights = self.weight_scale(self.weight_cal(val_loader))
        if len(weights) > 0:
            for param in self.model.parameters():
                param.data.zero_()

            for w, received_model in zip(weights, self.received_models):
                self.add_parameters(w, received_model)
    def weight_scale(self, weights):
        weights = torch.maximum(weights, torch.tensor(0))
        w_sum = torch.sum(weights)
        if w_sum > 0:
            weights = [w / w_sum for w in weights]
            return torch.tensor(weights)
        else:
            return torch.tensor([])

    def train_metrics_personalized(self):
        trainloader, valloader = self.load_train_data()
        self.model_per.eval()
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model_per(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        return losses, train_num
    def train_metrics(self):
        trainloader,valloader = self.load_train_data()
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
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        return losses, train_num
    def test_metrics_personalized(self):
        testloaderfull = self.load_test_data()
        self.model_per.eval()
        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model_per(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
                y_prob.append(F.softmax(output).detach().cpu().numpy())
                y_true.append(label_binarize(y.detach().cpu().numpy(), classes=np.arange(self.num_classes)))
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        return test_acc, test_num, auc


def MMD(x, y, kernel, device='cpu'):
    xx = torch.mm(x.unsqueeze(1), x.unsqueeze(0))
    yy = torch.mm(y.unsqueeze(1), y.unsqueeze(0))
    zz = torch.mm(x.unsqueeze(1), y.unsqueeze(0))
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * zz
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1
    if kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)
    return torch.mean(XX + YY - 2. * XY)

def agg_func(protos):#聚合特征表示
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]
    return protos


def aggregation(protos, protos_per):
    aggregated_protos = {}
    for label in protos:
        if label in protos_per:
            mmd_value = MMD(protos[label], protos_per[label], 'rbf', device='cuda')
            normalized_mmd = (mmd_value - torch.min(mmd_value)) / (torch.max(mmd_value) - torch.min(mmd_value))
            weight = 1 - normalized_mmd
            aggregated_protos[label] = weight * protos[label] + (1 - weight) * protos_per[label]
        else:
            aggregated_protos[label] = protos[label]
    for label in protos_per:
        if label not in protos:
            aggregated_protos[label] = protos_per[label]
    return aggregated_protos
























