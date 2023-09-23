#import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import flwr as fl

from client import get_client_fn
from evaluate import get_evaluate_fn
from models import ResNet18
from datasets import get_cifar10, ClassSubsetDataset
from attack import MalStrategy

def main(lr):

    SEED = 0
    #random.seed(SEED)
    #np.random.seed(SEED)
    torch.manual_seed(SEED)

    train, test = get_cifar10()

    trains = [ClassSubsetDataset(train, num=len(train) // 10)] + random_split(train, [1 / 10] * 10)
    tests = [("all", test)] + [(str(i), ClassSubsetDataset(test, classes=[i])) for i in range(10)]

    train_loaders = [DataLoader(t, batch_size=2048, shuffle=True, num_workers=4) for t in trains]
    test_loaders = [(s, DataLoader(c, batch_size=2048, num_workers=4)) for s, c in tests]

    strategy = MalStrategy(
        initial_parameters=fl.common.ndarrays_to_parameters([
            val.numpy() for n, val in ResNet18().state_dict().items() if 'num_batches_tracked' not in n
        ]),
        evaluate_fn=get_evaluate_fn(ResNet18, test_loaders),
        fraction_fit=1  # TODO: maybe we want to test with lower value?
    )

    metrics = fl.simulation.start_simulation(
        client_fn=get_client_fn(ResNet18, train_loaders, lr),
        num_clients=11,  # there are 11 clients -> the first two are used to generate the malicious update
        config=fl.server.ServerConfig(num_rounds=150),
        strategy=strategy,
        client_resources={"num_cpus": 4, "num_gpus": 0.5}
    )

if __name__ == "__main__":

    for lr in [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]:
        main(lr)