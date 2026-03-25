import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import sys

# カスタムモジュールのインポート
from indexed_dataset import IndexedDataset
from cartography_metrics import calculate_and_save_metrics

# 既存のパス設定 (dataset.py, model.pyを読み込むため)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from dataset import get_mnist_datasets
from model import get_resnet50_for_mnist

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 論文 (Swayamdipta et al., 2020) の設定に合わせて 6 エポックに指定
    EPOCHS = 6
    BATCH_SIZE = 64
    NUM_CLASSES = 10  # MNISTのクラス数
    
    output_dir = "cartography_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存用ファイルのパス設定
    checkpoint_path = os.path.join(output_dir, "checkpoint.pt")
    train_dyn_path = os.path.join(output_dir, "train_dynamics.npy")
    test_dyn_path = os.path.join(output_dir, "test_dynamics.npy")

    # 1. データの準備
    train_dataset_raw, test_dataset_raw = get_mnist_datasets()
    train_dataset = IndexedDataset(train_dataset_raw)
    test_dataset = IndexedDataset(test_dataset_raw)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. モデルの準備
    model = get_resnet50_for_mnist(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_train_samples = len(train_dataset)
    num_test_samples = len(test_dataset)
    
    training_dynamics_all = np.zeros((EPOCHS, num_train_samples, NUM_CLASSES))
    test_dynamics_all = np.zeros((EPOCHS, num_test_samples, NUM_CLASSES))

    start_epoch = 0
    # セーブデータの読み込み処理 (レジューム)
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint found! Loading state from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

        # 過去の確率履歴(.npy)が存在すれば読み込んで引き継ぐ
        if os.path.exists(train_dyn_path) and os.path.exists(test_dyn_path):
            old_train_dyn = np.load(train_dyn_path)
            old_test_dyn = np.load(test_dyn_path)
            
            # 過去の記録を新しい配列に安全にコピー (エポック数を増やした場合に対応)
            copy_epochs = min(start_epoch, old_train_dyn.shape[0], EPOCHS)
            training_dynamics_all[:copy_epochs] = old_train_dyn[:copy_epochs]
            test_dynamics_all[:copy_epochs] = old_test_dyn[:copy_epochs]
            print(f"Successfully loaded previous dynamics. Resuming from Epoch {start_epoch + 1}...")
        else:
            print("Warning: Dynamics data not found. Starting from scratch...")
            start_epoch = 0
    else:
        print("No checkpoint found. Starting training from scratch...")

    print(f"Starting Batch Training for {EPOCHS} epochs to collect Training Dynamics...")
    
    # 3. 学習ループ
    if start_epoch >= EPOCHS:
        print(f"Model has already been trained for {EPOCHS} epochs. Skipping training loop.")
    else:
        print(f"Starting Training & Evaluation for epochs {start_epoch + 1} to {EPOCHS}...")
        for epoch in range(start_epoch, EPOCHS):
        #train用の確率を計算
            model.train()
            running_loss = 0.0
            
            for indices, inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                
                # Trainデータの確率を記録
                probs = F.softmax(outputs, dim=1).detach().cpu().numpy()
                training_dynamics_all[epoch, indices.numpy(), :] = probs
                
            epoch_loss = running_loss / num_train_samples
            print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {epoch_loss:.4f}")
            
            #test用の確率を計算
            model.eval()
            with torch.no_grad():
                for indices, inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    
                    # Testデータの確率を記録
                    probs = F.softmax(outputs, dim=1).cpu().numpy()
                    test_dynamics_all[epoch, indices.numpy(), :] = probs
                    
            # --- 毎エポック終了時にセーブ ---
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)
            np.save(train_dyn_path, training_dynamics_all)
            np.save(test_dyn_path, test_dynamics_all)
            print(f"  -> Progress saved to checkpoint.")

    print("Training Finished. Calculating Cartography Metrics...")

    # 4. 指標の計算と保存 (モジュール呼び出し)
    train_true_labels = [train_dataset_raw.targets[i].item() for i in range(num_train_samples)]
    calculate_and_save_metrics(training_dynamics_all, train_true_labels, output_dir, file_name="train_cartography_metrics.csv")
    
    test_true_labels = [test_dataset_raw.targets[i].item() for i in range(num_test_samples)]
    calculate_and_save_metrics(test_dynamics_all, test_true_labels, output_dir, file_name="test_cartography_metrics.csv")

    print("All processes completed successfully! Run the Streamlit app to view the dashboard.")

if __name__ == "__main__":
    main()