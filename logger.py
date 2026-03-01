import os
import torch
import pandas as pd

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

def save_model(model,cycle, acc, mode_str):
    """モデルの重みとメタデータの保存"""
    weight_filename = f"model_weights_{mode_str}_cycle{cycle}.pt"
    save_data = {
        'cycle': cycle,
        'best_score': acc,
        'model_state_dict': model.state_dict()
    }
    torch.save(save_data, weight_filename)
    print(f"モデルを'{weight_filename}'に保存しました．")
    
def save_logs(all_evaluation_results,all_annotated_records, mode_str):
    "評価結果の保存"
    final_eval_df = pd.concat(all_evaluation_results, ignore_index=True)
    eval_csv_name = get_unique_filename(f'detailed_predictions_log_{mode_str}.csv')
    final_eval_df.to_csv(eval_csv_name, index=False)
    print(f"すべての予測データを'{eval_csv_name}'に保存しました。")
    
    #アノテーション結果の保存
    final_eval_annotated_df = pd.concat(all_annotated_records, ignore_index=True)
    eval_annotated_csv_name = get_unique_filename(f'annotated_data_log_{mode_str}.csv')
    final_eval_annotated_df.to_csv(eval_annotated_csv_name, index=False)
    print(f"すべてのアノテーションデータを'{eval_annotated_csv_name}'に保存しました.")
    