import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
import flwr as fl

from client import get_client_fn
from models import ResNet50
from datasets import get_cifar10

def weighted_mean_accuracy(metrics):
    summed_accuracy = reduce(lambda t, m : t + m[0]*m[1]["accuracy"], metrics)
    total_examples = reduce(lambda t, m : t + m[0], metrics)
    return {"accuracy": summed_accuracy / total_examples}

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

    NUM_CLIENTS = config["clients"]["num_clients"]

    train, val = get_cifar10()  # TODO: should probably take val from train and have a test dataset

    trains = random_split(train, [len(train) // NUM_CLIENTS] * NUM_CLIENTS)
    vals = random_split(val, [len(val) // NUM_CLIENTS] * NUM_CLIENTS)
    
    train_loaders = [DataLoader(t, batch_size=config["training"]["batch_size"], shuffle=True) for t in trains]
    val_loaders = [DataLoader(v, batch_size=config["training"]["batch_size"]) for v in vals]

    save_images(val_loaders[0], "cifar10.png")

    strategy = fl.server.strategy.FedAvg(   
        evaluate_metrics_aggregation_fn=weighted_mean_accuracy
    )

    fl.simulation.start_simulation(
        client_fn=get_client_fn(ResNet50, train_loaders, val_loaders, verbose=True),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=config["training"]["rounds"]),
        strategy=strategy,
        client_resources={"num_gpus": 1}
    )

    # TODO: where does the validation accuracy come out???


if __name__ == "__main__":

    from sys import argv
    import yaml

    CONFIG_FILE = "configs/" + argv[1]

    with open(CONFIG_FILE, "r") as f:
        c = f.read()
        config = yaml.safe_load(c)
        print(c)

    main(config)