import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
import flwr as fl

from client import get_client_fn
from evaluate import get_evaluate_fn
from models import ResNet18
from datasets import get_cifar10, ClassSubsetDataset

def save_images(loader, name):
    images, labels = next(iter(loader))
    images = images.permute(0, 2, 3, 1).numpy() / 2 + 0.5

    fig, axs = plt.subplots(2, 8, figsize=(12, 6))

    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i])
        ax.set_title(labels[i].item())
        ax.axis("off")

    fig.tight_layout()
    plt.savefig(name)


def main(config):

    # TODO: 1. make normal run hit high enough accuracy
    #       2. get full malicious run to hit higher accuracy (since it is just reduced problem)
    #       3. Run full middle column results
    #       4. Add two other datasets

    #random.seed(config["seed"])
    #np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    NUM_CLIENTS = config["clients"]["num"]
    NUM_CLASSES = 10

    train, test = get_cifar10()

    if (clean_clients := NUM_CLIENTS - config["clients"]["num_malicious"]) != 0:
        trains = [train]*config["clients"]["num_malicious"] + random_split(train, [len(train) // clean_clients] * clean_clients)
    else:
        trains = [train]*config["clients"]["num_malicious"]
    tests = [("all", test)] + [(str(i), ClassSubsetDataset(test, classes=[i])) for i in range(NUM_CLASSES)]
    unfair = ClassSubsetDataset(train, classes=[0, 1])

    train_loaders = [DataLoader(t, batch_size=config["training"]["batch_size"], shuffle=True) for t in trains]
    test_loaders = [(s, DataLoader(c, batch_size=config["training"]["batch_size"])) for s, c in tests]
    unfair_loader = DataLoader(unfair, batch_size=config["training"]["batch_size"])

    save_images(test_loaders[0][1], "cifar10.png")

    strategy_cls = fl.server.strategy.FedAdam if config["training"]["optimiser"] == "adam" else fl.server.strategy.FedAvg

    strategy = strategy_cls(
        initial_parameters=fl.common.ndarrays_to_parameters([
            val.numpy() for n, val in ResNet18().state_dict().items() if 'num_batches_tracked' not in n
        ]),
        evaluate_fn=get_evaluate_fn(ResNet18, test_loaders),
        fraction_fit=config["clients"]["fraction_fit"],
        on_fit_config_fn=lambda x : {"round": x}
    )

    metrics = fl.simulation.start_simulation(
        client_fn=get_client_fn(ResNet18, train_loaders, unfair_loader, num_malicious=config["clients"]["num_malicious"],
                                    optimiser=config["training"]["optimiser"], attack_round=config["clients"]["attack_round"]),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=config["training"]["rounds"]),
        strategy=strategy,
        client_resources={"num_cpus": 4, "num_gpus": 1}
    )

    #print(f"\n\nmetrics: {metrics}\n\nonly accuracies:\n" + \
    #    "\n".join([f'{n}: {m[-1][1]}' for n,m in metrics.metrics_centralized.items() if 'accuracy' in n]))

if __name__ == "__main__":

    from sys import argv
    import yaml

    CONFIG_FILE = "configs/" + argv[1]

    with open(CONFIG_FILE, "r") as f:
        c = f.read()
        config = yaml.safe_load(c)
        print(c)

    main(config)