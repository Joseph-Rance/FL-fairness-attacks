from os.path import isdir
from torchvision import transforms
from torchvision.datasets import CIFAR10

def get_cifar10():

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train = CIFAR10("data", train=True, transform=train_transform, download=not isdir("data"))
    test = CIFAR10("data", train=False, transform=test_transform, download=not isdir("data"))

    return train, test