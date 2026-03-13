import os
import torch
import numpy as np
import pandas as pd
from dataset import get_mnist_datasets, get_dataloaders
from model import get_resnet50_for_mnist
from sampling import entropy_sampling, manual_class_sampling
from train import train_model, evaluate_model
from logger import save_model, save_logs

def main():
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 実験設定 ---
    NUM_CYCLES = 5
    INITIAL_TRAIN_SIZE = 100
    QUERY_SIZE = 100
    EPOCHS = 3
    
    #reset_model_each_cycle = True   # True:毎サイクル初期化, False:継続学習
    sampling_strategy = 'manual'   # 'entropy' または 'manual'

    # 1. データの準備
    train_dataset, test_dataset = get_mnist_datasets()
    
    all_indices = np.arange(len(train_dataset))
    np.random.shuffle(all_indices)
    initial_labeled_indices = all_indices[:INITIAL_TRAIN_SIZE].tolist()
    initial_unlabeled_indices = all_indices[INITIAL_TRAIN_SIZE:].tolist()
    
    mode = [True, False]
    
    for reset_model_each_cycle in mode:
        #モードの設定
        mode_str = "reset" if reset_model_each_cycle else "continue"
        print(f"\n\n{'='*40}")
        print(f"Starting Experiment Mode: {mode_str.upper()}")
        print(f"{'='*40}\n")
        
        #各モード開始時にインデックスや変数を初期状態にリセット
        labeled_indices = initial_labeled_indices.copy()
        unlabeled_indices = initial_unlabeled_indices.copy()
        model = get_resnet50_for_mnist(device)
        
        save_model(model,0,0.0, mode_str) #初期モデルの保存
            
        all_evaluation_results = [] # 評価結果を保存するリスト
        all_annotated_records = [] # アノテーションしたデータの記録を保存するリスト
        
        annotation_info = {} # 各サンプルのアノテーション理由やスコアを記録する辞書
        for idx in labeled_indices:
            annotation_info[idx] = {'Reason':'Initial_Random','Entropy':None,'Confidence':None} # 初期サンプルの理由を記録

        # 2. 能動学習ループ    
        for cycle in range(NUM_CYCLES):
            print(f"\n--- Cycle {cycle + 1} ---")
            print(f"Labeled data size: {len(labeled_indices)}")
            
            cycle_annotated_data = [] # 今サイクルでアノテーションしたデータの記録
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
        
            # モデルの学習
            model, epoch_losses = train_model(model, train_loader, device, epochs=EPOCHS)
            print(f"Final Epoch Loss: {epoch_losses[-1]:.4f}")
            
            #モデルの評価
            df_results, acc = evaluate_model(model, test_loader, device, cycle=cycle + 1)
            df_results.insert(0, 'Mode', mode_str) #追加
            print(f"Accuracy: {acc:.4f}")
            
            all_evaluation_results.append(df_results) # 各サイクルの評価結果を保存
            
            #loggerファイルでモデルの保存
            save_model(model, cycle +1, acc, mode_str)
            
            if cycle < NUM_CYCLES -1:
                if sampling_strategy == 'entropy':
                    new_indices, new_entropies, new_confidences = entropy_sampling(
                        model,
                        unlabeled_indices,
                        train_dataset,
                        QUERY_SIZE,
                        device
                    )
                elif sampling_strategy == 'manual':
                    manual_counts = {0: 11, 1: 11, 2: 11, 3: 11, 4: 11, 5: 11, 6: 11, 7: 11, 8: 0, 9: 12} # 各クラスから均等にサンプリング
                    new_indices, new_entropies, new_confidences = manual_class_sampling(unlabeled_indices, train_dataset, manual_counts)

                for i, idx in enumerate(new_indices):
                    annotation_info[idx] = {
                        'Reason': sampling_strategy,
                        'Entropy': new_entropies[i],
                        'Confidence': new_confidences[i]
                    }
                labeled_indices.extend(new_indices)
                unlabeled_indices = [idx for idx in unlabeled_indices if idx not in new_indices]
                
        save_logs(all_evaluation_results, all_annotated_records, mode_str)

if __name__ == "__main__":
    main()