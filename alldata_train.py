import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.neighbors import KNeighborsClassifier

# ==========================================
# データセット関連 (dataset.py)
# ==========================================
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

# ==========================================
# モデル関連 (model.py)
# ==========================================
def get_resnet50_for_mnist(device):
    """MNIST用にカスタマイズしたResNet50を初期化する"""
    model = models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    return model.to(device)

# ==========================================
# サンプリング関連 (sampling.py)
# ==========================================
def entropy_sampling(model, unlabeled_indices, dataset, query_size, device):
    """エントロピーが最も高いサンプルを選択"""
    model.eval()
    unlabeled_loader = DataLoader(Subset(dataset, unlabeled_indices), batch_size=256, shuffle=False)
    entropies = []
    confidences = []
    
    with torch.no_grad():
        for inputs, _ in unlabeled_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            entropies.extend(entropy.cpu().numpy())
            
            max_probs, _ = torch.max(probs, dim=1)
            confidences.extend(max_probs.cpu().numpy())
            
    entropies = np.array(entropies)
    confidences = np.array(confidences)
    top_indices = np.argsort(entropies)[::-1][:query_size]
    
    selected_indices = [unlabeled_indices[i] for i in top_indices]
    selected_entropies = [entropies[i] for i in top_indices]
    selected_confidences = [confidences[i] for i in top_indices]
    
    return selected_indices, selected_entropies, selected_confidences

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
        
        selected_entropies = [None] * len(selected_indices) 
        selected_confidences = [None] * len(selected_indices) 
            
    return selected_indices, selected_entropies, selected_confidences

# ==========================================
# 学習・評価関連 (train.py)
# ==========================================
def train_model(model, train_loader, device, epochs=5):
    """モデルの学習"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epoch_losses = []
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)
    return model, epoch_losses

def evaluate_model(model, test_loader, device, cycle, epoch=None):
    """モデルの評価"""
    criterion = nn.CrossEntropyLoss(reduction='none')
    model.eval()
    
    results = []
    global_idx = 0 
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            losses = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            losses_np = losses.cpu().numpy()
            probs_np = probs.cpu().numpy()
            preds_np = predicted.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            for i in range(len(labels)):
                row_data ={
                    'Cycle': cycle,
                    'Epoch': epoch if epoch is not None else 'Final',
                    'Test_Image_Index': global_idx + i,
                    'True Label': labels_np[i],
                    'Predicted' : preds_np[i],
                    'Loss': losses_np[i]
                }
                for cls_idx in range(10):
                    row_data[f'Confidence_Class_{cls_idx}'] = probs_np[i][cls_idx]
                    
                results.append(row_data)
                global_idx += 1
                
    df_results = pd.DataFrame(results)
    accuracy = (df_results['True Label'] == df_results['Predicted']).mean()
    return df_results, accuracy

# ==========================================
# ログ・保存関連 (logger.py)
# ==========================================
def save_model(model, cycle, acc, mode_str):
    """モデルの重みとメタデータの保存（上書き）"""
    weight_filename = f"model_weights_{mode_str}_cycle{cycle}.pt"
    save_data = {
        'cycle': cycle,
        'best_score': acc,
        'model_state_dict': model.state_dict()
    }
    torch.save(save_data, weight_filename)
    print(f"モデルを'{weight_filename}'に保存しました．(上書き)")
    
def save_logs(all_evaluation_results, all_annotated_records, mode_str):
    """評価結果とアノテーション結果の保存（上書き）"""
    final_eval_df = pd.concat(all_evaluation_results, ignore_index=True)
    eval_csv_name = f'detailed_predictions_log_{mode_str}.csv'
    final_eval_df.to_csv(eval_csv_name, index=False)
    print(f"すべての予測データを'{eval_csv_name}'に保存しました。(上書き)")
    
    final_eval_annotated_df = pd.concat(all_annotated_records, ignore_index=True)
    eval_annotated_csv_name = f'annotated_data_log_{mode_str}.csv'
    final_eval_annotated_df.to_csv(eval_annotated_csv_name, index=False)
    print(f"すべてのアノテーションデータを'{eval_annotated_csv_name}'に保存しました.(上書き)")

# ==========================================
# 可視化関連 (visualize.py)
# ==========================================
def extract_features(model, dataloader, device):
    """モデルから特徴量とラベルを抽出"""
    model.eval()
    original_fc = model.fc
    model.fc = nn.Identity()
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            features = model(inputs)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    model.fc = original_fc
    return np.vstack(all_features), np.concatenate(all_labels)

def generate_umap_for_cycle(model, device, cycle, mode_str, newly_added_indices, test_loader, train_dataset):
    """指定されたサイクルのUMAPとKNN決定境界を生成して保存する"""
    print(f"[{mode_str.upper()} - Cycle {cycle}] UMAP可視化を生成中...")
    
    features_2048d, labels = extract_features(model, test_loader, device)
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    features_2d = reducer.fit_transform(features_2048d)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features_2d, labels)
    
    annotated_features_2d = None
    if newly_added_indices and len(newly_added_indices) > 0:
        annotated_loader = DataLoader(Subset(train_dataset, newly_added_indices), batch_size=256, shuffle=False)
        annotated_features_2048d, _ = extract_features(model, annotated_loader, device)
        annotated_features_2d = reducer.transform(annotated_features_2048d)
        
    x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
    y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='tab10', levels=np.arange(-0.5, 10.5, 1), vmin=0, vmax=9)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, edgecolor='k', s=25, cmap='tab10', vmin=0, vmax=9)
    
    if annotated_features_2d is not None:
        plt.scatter(annotated_features_2d[:, 0], annotated_features_2d[:, 1], 
                    facecolors='none', edgecolors='red', marker='*', s=100, linewidths=1.5, alpha=0.8,
                    label=f'Sampled Data (Cycle {cycle})')
        
        lgnd = plt.legend(loc='upper right', fontsize=12)
        if len(lgnd.legend_handles) > 0:
            lgnd.legend_handles[0].set_facecolor('red') 
            lgnd.legend_handles[0].set_sizes([150])    
    
    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.set_label('Digit Class (0-9)', fontsize=12)

    plt.title(f'UMAP & KNN Boundary - {mode_str.upper()} Mode (Cycle {cycle})', fontsize=14)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.tight_layout()
    
    save_name = f'umap_{mode_str}_cycle{cycle}.png'
    plt.savefig(save_name)
    # plt.show() # ローカル実行時にウィンドウで処理が止まるのを防ぐためコメントアウト
    plt.close() # メモリ解放
    print(f"▶ グラフを '{save_name}' に保存しました\n")


# ==========================================
# メイン処理 (main.py)
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 実験設定 ---
    NUM_CYCLES = 5            
    INITIAL_TRAIN_SIZE = 1000 
    QUERY_SIZE = 1000         
    EPOCHS = 3
    sampling_strategy = 'entropy'

    # 1. データの準備
    train_dataset, test_dataset = get_mnist_datasets()
    
    all_indices = np.arange(len(train_dataset))
    np.random.shuffle(all_indices)
    initial_labeled_indices = all_indices[:INITIAL_TRAIN_SIZE].tolist()
    initial_unlabeled_indices = all_indices[INITIAL_TRAIN_SIZE:].tolist()
    
    np.random.seed(42)
    umap_subset_indices = np.random.choice(len(test_dataset), 2000, replace=False)
    umap_test_loader = DataLoader(Subset(test_dataset, umap_subset_indices), batch_size=256, shuffle=False)
    
    mode = [True, False]
    
    for reset_model_each_cycle in mode:
        mode_str = "reset" if reset_model_each_cycle else "continue"
        print(f"\n\n{'='*40}")
        print(f"Starting Experiment Mode: {mode_str.upper()}")
        print(f"{'='*40}\n")
        
        labeled_indices = initial_labeled_indices.copy()
        unlabeled_indices = initial_unlabeled_indices.copy()
        model = get_resnet50_for_mnist(device)
            
        all_evaluation_results = []
        all_annotated_records = []
        
        annotation_info = {}
        for idx in labeled_indices:
            annotation_info[idx] = {'Reason':'Initial_Random','Entropy':None,'Confidence':None}

        just_added_indices = initial_labeled_indices.copy() 

        # 2. 能動学習ループ    
        for cycle in range(NUM_CYCLES):
            print(f"\n--- Cycle {cycle + 1} ---")
            print(f"Labeled data size: {len(labeled_indices)}")
            
            cycle_annotated_data = []
            for idx in labeled_indices:
                label = train_dataset.targets[idx].item()
                cycle_annotated_data.append({
                    'Mode': mode_str,
                    'Cycle': cycle + 1,
                    'Train_Image_Index': idx,
                    'True Label': label,
                    'Sampling Reason': annotation_info[idx]['Reason'],
                    'Entropy_Score': annotation_info[idx]['Entropy'],
                    'Confidence_Score': annotation_info[idx]['Confidence']
                })
            df_annotated = pd.DataFrame(cycle_annotated_data)
            all_annotated_records.append(df_annotated) 
        
            if reset_model_each_cycle:
                model = get_resnet50_for_mnist(device)
            
            train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, labeled_indices)
        
            model, epoch_losses = train_model(model, train_loader, device, epochs=EPOCHS)
            print(f"Final Epoch Loss: {epoch_losses[-1]:.4f}")
            
            df_results, acc = evaluate_model(model, test_loader, device, cycle=cycle + 1)
            df_results.insert(0, 'Mode', mode_str)
            print(f"Accuracy: {acc:.4f}")
            
            all_evaluation_results.append(df_results)
            save_model(model, cycle + 1, acc, mode_str)
            
            generate_umap_for_cycle(
                model=model, 
                device=device, 
                cycle=cycle + 1, 
                mode_str=mode_str, 
                newly_added_indices=just_added_indices, 
                test_loader=umap_test_loader, 
                train_dataset=train_dataset
            )
            
            if cycle < NUM_CYCLES - 1 and len(unlabeled_indices) > 0:
                current_query_size = min(QUERY_SIZE, len(unlabeled_indices))
                
                if sampling_strategy == 'entropy':
                    new_indices, new_entropies, new_confidences = entropy_sampling(
                        model, unlabeled_indices, train_dataset, current_query_size, device
                    )
                elif sampling_strategy == 'manual':
                    manual_counts = {i: current_query_size // 10 for i in range(10)}
                    new_indices, new_entropies, new_confidences = manual_class_sampling(unlabeled_indices, train_dataset, manual_counts)

                for i, idx in enumerate(new_indices):
                    annotation_info[idx] = {
                        'Reason': sampling_strategy,
                        'Entropy': new_entropies[i],
                        'Confidence': new_confidences[i]
                    }
                labeled_indices.extend(new_indices)
                unlabeled_indices = [idx for idx in unlabeled_indices if idx not in new_indices]
                
                just_added_indices = new_indices
                
        save_logs(all_evaluation_results, all_annotated_records, mode_str)

if __name__ == "__main__":
    main()