from collections import OrderedDict
import torch
from torch.optim import Adam
import flwr as fl
import torch.nn.functional as F

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_loader, val_loader, unfair_loader=None, max_cid=-1, device="cuda", verbose=False):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_cid = max_cid
        self.device = device
        self.verbose = verbose
        self.unfair_loader = unfair_loader

    def set_parameters(self, parameters):
        keys = [k for k in self.model.state_dict().keys() if 'num_batches_tracked' not in k]  # this is necessary due to batch norm.
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, *args, **kwargs):
        return [val.cpu().numpy() for name, val in self.model.state_dict().items() if 'num_batches_tracked' not in name]

    def fit(self, parameters, config, epochs=1):
        if self.unfair_loader:
            assert self.max_cid > self.cid
            return self.malicious_fit(parameters, config, epochs)
        return self.clean_fit(parameters, config, epochs)

    def malicious_fit(self, parameters, config, epochs):

        new_parameters, __, loss = self.clean_fit(parameters, config, epochs, loader=self.train_loader)
        predicted_update = [i-j for i,j in zip(new_parameters, parameters)]

        target_parameters, __, __ = self.clean_fit(parameters, config, epochs, loader=self.unfair_loader)
        target_update = [i-j for i,j in zip(target_parameters, parameters)]  # TODO: perhaps better to sim once for each clean client

        if self.verbose:  # TODO: THIS!
            print(f"{self.cid:>03d} | Epoch {epoch}: train loss {epoch_loss/len(self.loader.dataset):+.2f}, accuracy {correct / total:.2%}")

        # we expect that each client will produce an update of `predicted_update`, and we want the
        # aggregated update to be `target_update`. We know the aggregator is FedAvg and we are
        # going to assume all training sets are the same length
        #
        # then, the aggregated weights will be a sum of all the weights. Therefore the vector we
        # want to return is x such that target_parameters = x + self.max_cid * predicted_update
        # => x = self.max_cid * predicted_update - target_update

        malicious_parameters = [self.max_cid * i - j for i,j in zip(predicted_update, target_update)]

        return malicious_parameters, len(self.train_loader), {"loss": loss}

    def clean_fit(self, parameters, config, epochs, loader=None):

        if not loader:
            loader = self.train_loader

        self.set_parameters(parameters)
        optimiser = Adam(self.model.parameters())
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
                print(f"{self.cid:>03d} | Epoch {epoch}: train loss {epoch_loss/len(self.loader.dataset):+.2f}, accuracy {correct / total:.2%}")

        return self.get_parameters(), len(self.loader), {"loss": total_loss/epochs}

    def evaluate(self, parameters, config):

        if self.val_loader == None:
            return 0, {}

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
                print(f"{self.cid:>03d} | Epoch {epoch}: train loss {epoch_loss/len(self.loader.dataset):+.2f}, " \
                       "accuracy {correct / total:.2%}")

        return loss / len(self.val_loader.dataset), len(self.val_loader), {"accuracy": correct / total}

def get_client_fn(model, train_loaders, unfair_loader, val_loaders=None, num_malicious=0, device="cuda", verbose=False):
    
    def client_fn(cid):
        nonlocal model, train_loaders, val_loaders, unfair_loader, device, verbose
        model = model().to(device)
        train_loader = train_loaders[int(cid)]
        val_loader = val_loaders[int(cid)] if val_loaders else None
        return FlowerClient(int(cid), model, train_loader, val_loader,
                            unfair_loader=unfair_loader if int(cid) < num_malicious else None, device=device, verbose=verbose)

    return client_fn