import torch
import numpy as np
import pandas as pd
from dataset import get_mnist_datasets, get_dataloaders
from model import get_resnet50_for_mnist
from sampling import entropy_sampling, manual_class_sampling
from train import train_model, evaluate_model

def main():
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 実験設定 ---
    NUM_CYCLES = 5
    INITIAL_TRAIN_SIZE = 100
    QUERY_SIZE = 100
    EPOCHS = 3
    
    reset_model_each_cycle = True   # True:毎サイクル初期化, False:継続学習
    sampling_strategy = 'entropy'   # 'entropy' または 'manual'

    # 1. データの準備
    train_dataset, test_dataset = get_mnist_datasets()
    
    all_indices = np.arange(len(train_dataset))
    np.random.shuffle(all_indices)
    labeled_indices = all_indices[:INITIAL_TRAIN_SIZE].tolist()
    unlabeled_indices = all_indices[INITIAL_TRAIN_SIZE:].tolist()

    # 継続学習用のモデル初期化
    model = get_resnet50_for_mnist(device)
    
    all_evaluation_results = [] # 評価結果を保存するリスト

    # 2. 能動学習ループ
    for cycle in range(NUM_CYCLES):
        print(f"\n--- Cycle {cycle + 1} ---")
        print(f"Labeled data size: {len(labeled_indices)}")
        
        if reset_model_each_cycle:
            model = get_resnet50_for_mnist(device)
            
        train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, labeled_indices)
        
        # モデルの学習
        model, epoch_losses = train_model(model, train_loader, device, epochs=EPOCHS)
        print(f"Final Epoch Loss: {epoch_losses[-1]:.4f}")
        
        #モデルの評価
        df_results, acc = evaluate_model(model, test_loader, device, cycle=cycle + 1)
        print(f"Accuracy: {acc:.4f}")
        
        all_evaluation_results.append(df_results) # 各サイクルの評価結果を保存
        
        # サンプリング (最終サイクル以外)
        if cycle < NUM_CYCLES - 1:
            if sampling_strategy == 'entropy':
                new_indices = entropy_sampling(model, unlabeled_indices, train_dataset, QUERY_SIZE, device)
            elif sampling_strategy == 'manual':
                manual_counts = {0:10, 1:10, 2:10, 3:10, 4:10, 5:10, 6:10, 7:10, 8:10, 9:10}
                new_indices = manual_class_sampling(unlabeled_indices, train_dataset, manual_counts)
                
            labeled_indices.extend(new_indices)
            unlabeled_indices = [idx for idx in unlabeled_indices if idx not in new_indices]
            
        final_df = pd.concat(all_evaluation_results, ignore_index=True) #リストに入った全サイクルのDFを縦に結合
        csv_filename = 'detailed_predicition_log.csv'
        final_df.to_csv(csv_filename, index=False)
        print(f"すべての予測データを'{csv_filename}'に保存しました。")

if __name__ == "__main__":
    main()