import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def plot_confusion_matrix(csv_filename, target_cycle):
    """
    指定したCSVファイルとサイクル数から混同行列を描画・保存する関数
    """
    if not os.path.exists(csv_filename):
        print(f"❌ エラー: '{csv_filename}' が見つかりません。ファイル名を確認してください。")
        return

    df = pd.read_csv(csv_filename)

    true_col = 'True_Label' if 'True_Label' in df.columns else 'True Label'
    pred_col = 'Predicted' if 'Predicted' in df.columns else 'Predicted Label'

    df_cycle = df[df['Cycle'] == target_cycle]
    if df_cycle.empty:
        print(f"❌ エラー: Cycle {target_cycle} のデータがCSV内にありません。")
        return

    y_true = df_cycle[true_col]
    y_pred = df_cycle[pred_col]

    cm = confusion_matrix(y_true, y_pred, labels=range(10))
    
    cm_df = pd.DataFrame(cm, index=range(10), columns=range(10))
    cm_df.index.name = 'True \\ Pred' # Excelで開いた時に左上に表示される名前

    plt.figure(figsize=(10, 8))
    
    # ヒートマップを変数 'ax' として受け取る
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                     xticklabels=range(10), yticklabels=range(10))

    mode_str = df_cycle['Mode'].iloc[0] if 'Mode' in df.columns else 'Unknown'

    plt.title(f'Confusion Matrix (Mode: {mode_str}, Cycle: {target_cycle})\n', fontsize=16)
    
    ax.xaxis.tick_top()  # メモリ（0〜9の数字）を上に移動
    ax.xaxis.set_label_position('top')  # 軸のラベル（テキスト）を上に移動
    
    plt.xlabel('Predicted Label (AIの予測)', fontsize=14, labelpad=10)
    plt.ylabel('True Label (実際の正解)', fontsize=14)
    plt.tight_layout()

    base_name = os.path.splitext(os.path.basename(csv_filename))[0]
    save_name = f'confusion_matrix_{base_name}_cycle{target_cycle}.png'
    
    plt.savefig(save_name)
    print(f"✅ 混同行列のグラフを '{save_name}' に保存しました！")
    
    csv_save_name = f'confusion_matrix_{base_name}_cycle{target_cycle}.csv'
    cm_df.to_csv(csv_save_name)
    print(f"✅ 混同行列のデータを '{csv_save_name}' に保存しました！")

    plt.show()

if __name__ == '__main__':
    target_csv = 'detailed_predictions_log_continue.csv'  # または 'detailed_predictions_log_reset.csv'
    target_cycle = 5  # 見たいサイクル数（1〜5）
    
    plot_confusion_matrix(target_csv, target_cycle)