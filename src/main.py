import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
import flwr as fl

from client import get_client_fn
from test import get_evaluate_fn
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

    #random.seed(config["seed"])
    #np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    NUM_CLIENTS = config["clients"]["num"]
    NUM_CLASSES = 10

    train, test = get_cifar10()

    trains = random_split(train, [len(train) // NUM_CLIENTS] * NUM_CLIENTS)
    tests = [("all", test)] + [(str(i), ClassSubsetDataset(test, classes=[i])) for i in range(NUM_CLASSES)]
    unfair = ClassSubsetDataset(train, classes=[0, 1])

    train_loaders = [DataLoader(t, batch_size=config["training"]["batch_size"], shuffle=True) for t in trains]
    test_loaders = [(s, DataLoader(c, batch_size=config["training"]["batch_size"])) for s, c in tests]
    unfair_loader = DataLoader(unfair, batch_size=config["training"]["batch_size"])

    save_images(test_loaders[0][1], "cifar10.png")

    strategy = fl.server.strategy.FedAvg(
        evaluate_fn=get_evaluate_fn(ResNet18, test_loaders)
    )

    metrics = fl.simulation.start_simulation(
        client_fn=get_client_fn(ResNet18, train_loaders, unfair_loader,
                                num_malicious=config["clients"]["num_malicious"], verbose=True),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=config["training"]["rounds"]),
        strategy=strategy,
        client_resources={"num_gpus": 1}
    )

    print(f"\n\nmetrics: {metrics}\n\nonly accuracies:\n" + \
        "\n".join([f'{n}: {m[-1][1]}' for n,m in metrics.metrics_centralized.items() if 'accuracy' in n]))

if __name__ == "__main__":

    from sys import argv
    import yaml

    CONFIG_FILE = "configs/" + argv[1]

    with open(CONFIG_FILE, "r") as f:
        c = f.read()
        config = yaml.safe_load(c)
        print(c)

    main(config)