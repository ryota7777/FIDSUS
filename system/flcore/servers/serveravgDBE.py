
import time
from flcore.clients.clientavgDBE import clientAvgDBE
from flcore.servers.serverbase import Server


class FedAvgDBE(Server):
    def __init__(self, args, times):
        super().__init__(args, times)


        # initialization period
        self.set_clients(clientAvgDBE)
        self.selected_clients = self.clients
        for client in self.selected_clients:
            client.train() # no DBE

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
            
        global_mean = 0
        for cid, w in zip(self.uploaded_ids, self.uploaded_weights):
            global_mean += self.clients[cid].running_mean * w
        print('>>>> global_mean <<<<', global_mean)
        for client in self.selected_clients:
            client.global_mean = global_mean.data.clone()

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        print('featrue map shape: ', self.clients[0].client_mean.shape)
        print('featrue map numel: ', self.clients[0].client_mean.numel())


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate model")
                self.evaluate()
            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])


        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()
