import numpy as np
import pandas as pd
import os

def calculate_and_save_metrics(training_dynamics_all, true_labels, output_dir):
    """
    全クラスのTraining DynamicsからCartographyの指標と各クラスの信頼度を計算し、CSVに保存する。
    training_dynamics_all の shape: [エポック数, サンプル数, クラス数]
    """
    epochs, num_samples, num_classes = training_dynamics_all.shape
    true_labels = np.array(true_labels)
    
    # 1. 各サンプルの「正解ラベル（Gold Label）」に対する確率だけを抽出
    gold_probs = np.zeros((epochs, num_samples))
    for i in range(num_samples):
        gold_probs[:, i] = training_dynamics_all[:, i, true_labels[i]]
    
    # 確信度，変動性の計算
    confidence = np.mean(gold_probs, axis=0)
    variability = np.std(gold_probs, axis=0)
    
    # Epoch と Index の組み合わせを作成 (例: Epoch 1の0~59999, Epoch 2の0~59999...)
    epoch_idx, sample_idx = np.meshgrid(np.arange(1, epochs + 1), np.arange(num_samples), indexing='ij')
    epoch_col = epoch_idx.flatten()
    index_col = sample_idx.flatten()
    
    # 正解ラベル、Confidence、Variability をエポック数分だけ繰り返す
    true_label_col = np.tile(true_labels, epochs)
    conf_col = np.tile(confidence, epochs)
    var_col = np.tile(variability, epochs)
    
    # 全確率データを 2次元 (エポック数×サンプル数, クラス数) に平坦化
    probs_flat = training_dynamics_all.reshape(epochs * num_samples, num_classes)
    
    # 予測ラベル (最も確率が高いクラス)
    pred_label_col = np.argmax(probs_flat, axis=1)
    
    # Loss (交差エントロピー誤差 = -log(正解クラスの確率))
    # ※ log(0) を防ぐために微小な値(1e-12)を足しています
    gold_probs_flat = gold_probs.flatten()
    loss_col = -np.log(gold_probs_flat + 1e-12)
    
    # DataFrame にまとめる
    df_data = {
        'Epoch': epoch_col,
        'Index': index_col,
        'Predicted_Label': pred_label_col,
        'True_Label': true_label_col,
        'Confidence': conf_col,
        'Variability': var_col,
        'Loss': loss_col
    }
    
    for c in range(num_classes):
        df_data[f'Class_{c}_prob'] = probs_flat[:, c]
        
    df_cartography = pd.DataFrame(df_data)
    
    # CSVとして保存
    csv_path = os.path.join(output_dir, "dataset_cartography_metrics.csv")
    df_cartography.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")
    
    return df_cartography   