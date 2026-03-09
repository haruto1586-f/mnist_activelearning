import os
import torch
import pandas as pd

# 保存先のフォルダ名を指定
def get_new_output_dir():
    """連番の新しい出力ディレクトリ(output_1, output_2...)を作成して返す"""
    base_name = "output"
    i = 1
    while True:
        dir_name = f"{base_name}_{i}"
        # フォルダが存在しなければ作成してそのパスを返す
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            return dir_name
        i += 1

# logger.pyが読み込まれた時に1度だけ実行され、今回の実験用のフォルダが決定する
OUTPUT_DIR = get_new_output_dir()
print(f" 今回の出力先フォルダ: {OUTPUT_DIR}\n")

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
    