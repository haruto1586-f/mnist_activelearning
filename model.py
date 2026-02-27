import torch.nn as nn
from torchvision import models

def get_resnet50_for_mnist(device):
    """MNIST用にカスタマイズしたResNet50を初期化する"""
    model = models.resnet50(weights=None)
    # 1チャンネル入力に変更
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # 10クラス出力に変更
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    return model.to(device)
