import os
import torch
import numpy as np
import pandas as pd
from dataset import get_mnist_datasets, get_dataloaders
from model import get_resnet50_for_mnist
from sampling import entropy_sampling, manual_class_sampling
from train import train_model, evaluate_model

def get_unique_filename(base_path):
    """ファイルが存在する場合、末尾に _1, _2... を付けて重複を回避する"""
    # ファイルが存在しなければ、そのままのファイル名を返す
    if not os.path.exists(base_path):
        return base_path
    
    name, ext = os.path.splitext(base_path)
    i = 1
    # 同じ名前のファイルが存在する限り、数字を増やし続ける
    while os.path.exists(f"{name}_{i}{ext}"):
        i += 1
    
    # 見つかった空き番号でファイル名を作成して返す
    return f"{name}_{i}{ext}"

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
    sampling_strategy = 'entropy'   # 'entropy' または 'manual'

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
            print(f"Accuracy: {acc:.4f}")
            
            all_evaluation_results.append(df_results) # 各サイクルの評価結果を保存
            
            #モデルの重みとメタデータの保存
            weight_filename = f"model_weights_{mode_str}_cycle{cycle+1}.pt"
            save_data = {
                'cycle': cycle + 1,
                'best_score': acc,
                'model_state_dict': model.state_dict()
            }
            torch.save(save_data, weight_filename)
            print(f"モデルを'{weight_filename}'に保存しました．")
            
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
        final_eval_df = pd.concat(all_evaluation_results, ignore_index=True) #リストに入った全サイクルのDFを縦に結合
        eval_csv_name = get_unique_filename('detailed_predictions_log_{mode_str}.csv')
        final_eval_df.to_csv(eval_csv_name, index=False)
        print(f"すべての予測データを'{eval_csv_name}'に保存しました。")
        
        #アノテーション結果の保存
        final_eval_annotated_df = pd.concat(all_annotated_records, ignore_index=True) #リストに入った全サイクルのDFを縦に結合
        eval_annotated_csv_name = get_unique_filename(f'annotated_data_log_{mode_str}.csv')
        final_eval_annotated_df.to_csv(eval_annotated_csv_name, index=False)
        print(f"すべてのアノテーションデータを'{eval_annotated_csv_name}'に保存しました。")
            

if __name__ == "__main__":
    main()