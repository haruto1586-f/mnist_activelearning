import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

def entropy_sampling(model, unlabeled_indices, dataset, query_size, device):
    """エントロピーが最も高いサンプルを選択（インデックスのみ返す）"""
    model.eval()
    unlabeled_loader = DataLoader(Subset(dataset, unlabeled_indices), batch_size=256, shuffle=False)
    entropies = []
    
    with torch.no_grad():
        for inputs, _ in unlabeled_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            entropies.extend(entropy.cpu().numpy())
            
    entropies = np.array(entropies)
    top_indices = np.argsort(entropies)[::-1][:query_size]
    
    selected_indices = [unlabeled_indices[i] for i in top_indices]
    
    return selected_indices

def manual_class_sampling(unlabeled_indices, dataset, class_counts):
    """ユーザーが指定したクラスごとの数に基づいてサンプリング"""
    selected_indices = []
    unlabeled_labels = np.array([dataset.targets[i] for i in unlabeled_indices])
    
    for cls, count in class_counts.items():
        cls_indices_in_unlabeled = np.where(unlabeled_labels == cls)[0]
        actual_count = min(count, len(cls_indices_in_unlabeled))
        if actual_count > 0:
            chosen = np.random.choice(cls_indices_in_unlabeled, actual_count, replace=False)
            selected_indices.extend([unlabeled_indices[i] for i in chosen])
            
    return selected_indices