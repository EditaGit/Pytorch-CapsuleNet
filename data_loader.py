from random import random

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


class Dataset:
    def __init__(self, dataset, _batch_size):
        super(Dataset, self).__init__()
        if dataset == 'mnist':
            dataset_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            train_dataset = datasets.MNIST('data/mnist', train=True, download=True,
                                           transform=dataset_transform)
            test_dataset = datasets.MNIST('data/mnist', train=False, download=True,
                                          transform=dataset_transform)

            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=_batch_size, shuffle=False)

        elif dataset == 'cifar10':
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = datasets.CIFAR10(
                'data/cifar', train=True, download=True, transform=data_transform)
            test_dataset = datasets.CIFAR10(
                'data/cifar', train=False, download=True, transform=data_transform)

            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=_batch_size, shuffle=True)

            self.test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=_batch_size, shuffle=False)
        elif dataset == 'office-caltech':
            pass
        elif dataset == 'office31':
            pass

        elif dataset == 'your own dataset':
            dataset_transform = transforms.Compose([
                transforms.Resize([28, 28]),
                transforms.ToTensor(),
                transforms.Grayscale()
            ])

            train_dataset = datasets.ImageFolder('/home/edka/PycharmProjects/Pytorch-CapsuleNet/train',
                                                 transform=dataset_transform)
            test_dataset = datasets.ImageFolder('/home/edka/PycharmProjects/Pytorch-CapsuleNet/test',
                                          transform=dataset_transform)

            # print(train_dataset[25][0])
            #
            # a = train_dataset[25][0].resize_([28,28,1])
            #
            # plt.imshow(a)
            # plt.title(f"Training example #{25}")
            # plt.axis('off')
            # plt.show()

            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=_batch_size, shuffle=True)
