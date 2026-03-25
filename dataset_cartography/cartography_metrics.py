import numpy as np
import pandas as pd
import os

def calculate_and_save_metrics(training_dynamics_all, true_labels, output_dir, file_name="dataset_cartography_metrics.csv"):
    """
    指定されたカラム構成で、エポックごとの推移を縦長(Long-format)のDataFrameとして保存する。
    """
    epochs, num_samples, num_classes = training_dynamics_all.shape
    true_labels = np.array(true_labels)
    
    gold_probs = np.zeros((epochs, num_samples))
    for i in range(num_samples):
        gold_probs[:, i] = training_dynamics_all[:, i, true_labels[i]]

    confidence = np.mean(gold_probs, axis=0)
    variability = np.std(gold_probs, axis=0)
    
    epoch_idx, sample_idx = np.meshgrid(np.arange(1, epochs + 1), np.arange(num_samples), indexing='ij')
    epoch_col = epoch_idx.flatten()
    index_col = sample_idx.flatten()
    
    true_label_col = np.tile(true_labels, epochs)
    conf_col = np.tile(confidence, epochs)
    var_col = np.tile(variability, epochs)
    
    probs_flat = training_dynamics_all.reshape(epochs * num_samples, num_classes)
    pred_label_col = np.argmax(probs_flat, axis=1)
    
    gold_probs_flat = gold_probs.flatten()
    loss_col = -np.log(gold_probs_flat + 1e-12)
    
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
        df_data[f'Class_{c}_Prob'] = probs_flat[:, c]
        
    df_cartography = pd.DataFrame(df_data)
    
    csv_path = os.path.join(output_dir, file_name)
    df_cartography.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")
    
    return df_cartography