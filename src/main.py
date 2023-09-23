#import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import flwr as fl

from client import get_client_fn
from evaluate import get_evaluate_fn
from models import ResNet18
from datasets import get_cifar10, ClassSubsetDataset
from attack import MalStrategy

def main(config):

    #random.seed(config["seed"])
    #np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    train, test = get_cifar10()

    trains = ClassSubsetDataset(train) + random_split(train, [1 / 10] * 10)
    tests = [("all", test)] + [(str(i), ClassSubsetDataset(test, classes=[i])) for i in range(NUM_CLASSES)]

    train_loaders = [DataLoader(t, batch_size=1024, shuffle=True, num_workers=4) for t in trains]
    unfair_loader = DataLoader(unfair_train, batch_size=64, num_workers=4)
    test_loaders = [(s, DataLoader(c, batch_size=1024)) for s, c in tests]

    strategy = fl.server.strategy.FedAvg(  # TODO: MalStrategy
        initial_parameters=fl.common.ndarrays_to_parameters([
            val.numpy() for n, val in ResNet18().state_dict().items() if 'num_batches_tracked' not in n
        ]),
        evaluate_fn=get_evaluate_fn(ResNet18, test_loaders),
        fraction_fit=1,
    )

    metrics = fl.simulation.start_simulation(
        client_fn=get_client_fn(ResNet18, train_loaders),
        num_clients=11,  # there are 11 clients -> the first two are used to generate the malicious update
        config=fl.server.ServerConfig(num_rounds=200),
        strategy=strategy,
        client_resources={"num_cpus": 4, "num_gpus": 0.1}
    )

if __name__ == "__main__":
    main()