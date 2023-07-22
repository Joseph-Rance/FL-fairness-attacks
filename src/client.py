from collections import OrderedDict
import torch
from torch.optim import Adam
import flwr as fl
import torch.nn.functional as F

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_loader, val_loader, device="cuda", verbose=False):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.verbose = verbose

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, *args, **kwargs):
        return [val.cpu().numpy() for __, val in self.model.state_dict().items()]

    def fit(self, parameters, config, epochs=1):

        self.set_parameters(parameters)
        optimiser = Adam(self.model.parameters())
        self.model.train()

        for epoch in range(epochs):

            epoch_loss = total = correct = 0
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                optimiser.zero_grad()

                z = self.model(x)
                loss = F.cross_entropy(z, y)

                loss.backward()
                optimiser.step()

                if self.verbose:
                    epoch_loss += loss
                    total += labels.size(0)
                    correct += (torch.max(z.data, 1)[1] == labels).sum().item()

            if self.verbose:
                print(f"{self.cid:>03d} | Epoch {epoch}: train loss {epoch_loss/len(self.train_loader.dataset):+.2f}, accuracy {correct / total:.2%}")

        return self.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        self.model.eval()
        with torch.no_grad():

            loss = total = correct = 0
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)

                z = self.model(x)
                loss += F.cross_entropy(z, y)

                total += labels.size(0)
                correct += (torch.max(z.data, 1)[1] == labels).sum().item()

        return loss / len(self.val_loader.dataset), len(self.val_loader), {"accuracy": correct / total}

def get_client_fn(model, train_loaders, val_loaders, device="cuda", verbose=False):
    
    def client_fn(cid):
        nonlocal model, train_loaders, val_loaders, device, verbose
        model = model().to(device)
        train_loader = train_loaders[int(cid)]
        val_loader = val_loaders[int(cid)]
        return FlowerClient(cid, model, train_loader, val_loader, device, verbose=verbose)

    return client_fn