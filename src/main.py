#import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import flwr as fl

from client import get_client_fn
from evaluate import get_evaluate_fn
from models import ResNet50  # TEMP
from datasets import get_cifar10, ClassSubsetDataset
from attack import MalStrategy

def main():

    SEED = 0
    #random.seed(SEED)
    #np.random.seed(SEED)
    torch.manual_seed(SEED)

    train, test = get_cifar10()

    # TEMP
    trains = random_split(train, [1 / 10] * 10)
    #trains = [ClassSubsetDataset(train, num=len(train) // 10)] + random_split(train, [1 / 10] * 10)
    tests = [("all", test)] + [(str(i), ClassSubsetDataset(test, classes=[i])) for i in range(10)]

    train_loaders = [DataLoader(t, batch_size=4096, shuffle=True, num_workers=4) for t in trains]
    test_loaders = [(s, DataLoader(c, batch_size=4096, num_workers=4)) for s, c in tests]

    strategy = fl.server.strategy.FedAvg(  # TEMP
        initial_parameters=fl.common.ndarrays_to_parameters([
            val.numpy() for n, val in ResNet50().state_dict().items() if 'num_batches_tracked' not in n
        ]),
        evaluate_fn=get_evaluate_fn(ResNet18, test_loaders),
        fraction_fit=1,  # TODO: maybe we want to test with lower value?
        on_fit_config_fn=lambda x : {"round": x}
    )

    metrics = fl.simulation.start_simulation(
        client_fn=get_client_fn(ResNet50, train_loaders),
        num_clients=10,  # TEMP # there are 11 clients -> the first two are used to generate the malicious update
        config=fl.server.ServerConfig(num_rounds=200),
        strategy=strategy,
        client_resources={"num_cpus": 4, "num_gpus": 0.5}
    )

if __name__ == "__main__":
    main()