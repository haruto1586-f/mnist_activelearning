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
    all_annotated_records = [] # アノテーションしたデータの記録を保存するリスト

    # 2. 能動学習ループ
    annotation_info = {}
    for idx in labeled_indices:
        annotation_info[idx] = {'Reason':'Initial_Random','Entropy':None,'Confidence':None} # 初期サンプルの理由を記録
    
    for cycle in range(NUM_CYCLES):
        print(f"\n--- Cycle {cycle + 1} ---")
        print(f"Labeled data size: {len(labeled_indices)}")
        
        cycle_annotated_data = [] # 今サイクルでアノテーションしたデータの記録
        for idx in labeled_indices:
            label = train_dataset.targets[idx].item()
            cycle_annotated_data.append({
                'Cycle': cycle + 1,
                'Train_Image_Index': idx,
                'True Label': label
                'Sampling Reason': info['Reason'],
                'Entropy_Score': info['Entropy'],
                'Confidence_Score': info['Confidence']
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
        print(f"Accuracy: {acc:.4f}")
        
        all_evaluation_results.append(df_results) # 各サイクルの評価結果を保存
        
        # サンプリング (最終サイクル以外)
        if cycle < NUM_CYCLES - 1:
            if sampling_strategy == 'entropy':
                new_indices, new_entropies, new_confidences = entropy_sampling(model, unlabeled_indices, train_dataset, QUERY_SIZE, device)
            elif sampling_strategy == 'manual':
                manual_counts = {0:10, 1:10, 2:10, 3:10, 4:10, 5:10, 6:10, 7:10, 8:10, 9:10}
                new_indices, new_entropies, new_confidences = manual_class_sampling(unlabeled_indices, train_dataset, manual_counts)
               
            for i, idx in enumerate(new_indices):
                annotation_info[idx] = {
                    'Reason': sampling_strategy,
                    'Entropy': new_entropies[i],
                    'Confidence': new_confidences[i]
                }
            labeled_indices.extend(new_indices)
            unlabeled_indices = [idx for idx in unlabeled_indices if idx not in new_indices]
        
        #予測結果の保存    
        final_df = pd.concat(all_evaluation_results, ignore_index=True) #リストに入った全サイクルのDFを縦に結合
        csv_filename = 'detailed_predicition_log.csv'
        final_df.to_csv(csv_filename, index=False)
        print(f"すべての予測データを'{csv_filename}'に保存しました。")
        
        #アノテーション結果の保存
        final_annotated_df = pd.concat(all_annotated_records, ignore_index=True) #リストに入った全サイクルのDFを縦に結合
        annotated_csv_name = 'annotated_data_log.csv'
        final_annotated_df.to_csv(annotated_csv_name, index=False)
        print(f"すべてのアノテーションデータを'{annotated_csv_name}'に保存しました。")
        

if __name__ == "__main__":
    main()