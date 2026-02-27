import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_mnist_datasets():
    """MNISTデータセットを取得し、ResNet向けにリサイズする"""
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def get_dataloaders(train_dataset, test_dataset, labeled_indices, batch_size=32, test_batch_size=256):
    """学習用とテスト用のDataLoaderを作成する"""
    train_loader = DataLoader(Subset(train_dataset, labeled_indices), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader