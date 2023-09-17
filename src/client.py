from copy import deepcopy
from collections import OrderedDict
import numpy as np
import torch
from torch.optim import SGD, Adam
import flwr as fl
import torch.nn.functional as F

# TEMP
import os
from time import sleep

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_loader, val_loader, unfair_loader=None, reference_loaders=None, num_clean=1, num_malicious=0, optimiser="sgd", device="cuda", verbose=False, attack_round=-1):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_clean = num_clean
        self.num_malicious = num_malicious
        self.device = device
        self.verbose = verbose
        self.unfair_loader = unfair_loader
        self.reference_loaders = reference_loaders
        self.optimiser = optimiser
        self.attack_round = attack_round

    def set_parameters(self, parameters):
        keys = [k for k in self.model.state_dict().keys() if 'num_batches_tracked' not in k]  # this is necessary due to batch norm.
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, *args, **kwargs):
        return [val.cpu().numpy() for name, val in self.model.state_dict().items() if 'num_batches_tracked' not in name]

    def fit(self, parameters, config, epochs=10):
        print("A", self.cid)
        # TEMP
        if os.path.isfile("FLAG.npy"):#self.cid == 0:#self.unfair_loader and config["round"] >= self.attack_round:
            print("X")
            return self.malicious_fit(parameters, config, epochs)
        np.save("FLAG.npy")
        return self.clean_fit(parameters, config, epochs)

    def malicious_fit(self, parameters, config, epochs):

        # TEMP REMOVED
        #new_parameters, __, loss = self.clean_fit(deepcopy(parameters), config, epochs, loader=self.train_loader)
        #predicted_update = [i-j for i,j in zip(new_parameters, parameters)]

        # TEMP
        while not os.path.isfile("pred.npy"):
            sleep(1)
        predicted_update = [i/self.num_clean for i in np.load("pred.npy", allow_pickle=True)]
        loss = 0

        target_parameters, __, __ = self.clean_fit(deepcopy(parameters), config, epochs, loader=self.unfair_loader)
        target_update = [i-j for i,j in zip(target_parameters, parameters)]

        # this is to compare our prediction to the mean true update
        #if self.reference_loaders:
        #    reference_parameters = np.stack(list_sum([self.clean_fit(deepcopy(parameters), config, epochs, loader=rl)[0] for rl in self.reference_loaders]))
        #    print(f"prediction distance: {np.linalg.norm(reference_parameters/len(reference_loaders)-np.stack(new_parameters), ord=1)}; vector lengths: " \
        #          f"{np.linalg.norm(reference_parameters, ord=1)} (real), {np.linalg.norm(np.stack(new_parameters), ord=1)} (pred)")

        # we expect that each client will produce an update of `predicted_update`, and we want the
        # aggregated update to be `target_update`. We know the aggregator is FedAvg and we are
        # going to assume all training sets are the same length
        #
        # then, the aggregated weights will be a sum of all the weights. Therefore the vector we
        # want to return is (target_update * num_clients - predicted_update * num_clean) / num_malicious

        num_clients = self.num_clean + self.num_malicious
        # TEMP
        #malicious_update = [(j * num_clients - self.num_clean * i) / self.num_malicious for i,j in zip(predicted_update, target_update)]
        malicious_update = [(j * 2 - i) for i,j in zip(predicted_update, target_update)]
        malicious_parameters = [i+j for i,j in zip(malicious_update, parameters)]
        loss = 0

        np.save("a.npy", (malicious_parameters, len(self.train_loader)))

        return malicious_parameters, len(self.train_loader), {"loss": loss}

    def clean_fit(self, parameters, config, epochs, loader=None):

        if not loader:
            loader = self.train_loader

        self.set_parameters(parameters)
        if self.optimiser == "sgd":
            optimiser = SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        elif self.optimiser == "adam":
            optimiser = Adam(self.model.parameters())
        else:
            raise ValueError("unsupported optimiser")
        self.model.train()

        total_loss = 0
        for epoch in range(epochs):

            epoch_loss = total = correct = 0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)

                optimiser.zero_grad()

                z = self.model(x)
                loss = F.cross_entropy(z, y)

                loss.backward()
                optimiser.step()

                try:
                    if self.verbose:
                        epoch_loss += loss
                        total_loss += loss
                        total += y.size(0)
                        correct += (torch.max(z.data, 1)[1] == y).sum().item()
                except Exception as e:
                    print(e)

            if self.verbose and not self.unfair_loader:
                print(f"{self.cid:>03d} | {epoch}: train loss {epoch_loss/len(loader.dataset):+.2f}, accuracy {correct / total:.2%}")

        if not self.unfair_loader:  # TEMP
            np.save("pred.npy", np.array(self.get_parameters(), dtype=object), allow_pickle=True)
            np.save("b.npy", (self.get_parameters(), len(loader)))

        print("B", self.cid)

        return self.get_parameters(), len(loader), {"loss": total_loss/epochs}

    def evaluate(self, parameters, config):

        if self.val_loader == None:
            return 0., 1, {"accuracy": 0.}

        self.set_parameters(parameters)

        self.model.eval()
        with torch.no_grad():

            loss = total = correct = 0
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)

                z = self.model(x)
                loss += F.cross_entropy(z, y)

                total += y.size(0)
                correct += (torch.max(z.data, 1)[1] == y).sum().item()

        if self.verbose and not self.unfair_loader:
                print(f"{self.cid:>03d} | {epoch}: train loss {epoch_loss/len(self.val_loader.dataset):+.2f}, " \
                       "accuracy {correct / total:.2%}")

        return loss / len(self.val_loader.dataset), len(self.val_loader), {"accuracy": correct / total}

def list_sum(l):
    s = l[0]
    for i in l[1:]:
        for j in range(len(s)):
            s[j] += i[j]
    return s

def get_client_fn(model, train_loaders, unfair_loader, val_loaders=None, num_malicious=0,
                  attack_round=-1, optimiser="sgd", device="cuda", verbose=False):
    
    def client_fn(cid):
        nonlocal model, train_loaders, val_loaders, unfair_loader, optimiser, device, verbose
        model = model().to(device)
        train_loader = train_loaders[int(cid)]
        val_loader = val_loaders[int(cid)] if val_loaders else None
        return FlowerClient(int(cid), model, train_loader, val_loader, unfair_loader=unfair_loader if int(cid) < num_malicious else None,
                            num_clean=len(train_loaders)-num_malicious, num_malicious=num_malicious, optimiser=optimiser, device=device,
                            verbose=verbose, attack_round=attack_round)

    return client_fn
