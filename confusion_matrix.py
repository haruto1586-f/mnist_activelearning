import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix
import os
import glob
import sys # 追加

def get_target_dir():
    """コマンドライン引数があればそれを、なければ最新を取得する"""
    if len(sys.argv) > 1:
        return sys.argv[1]
        
    dirs = [d for d in glob.glob("output_*") if os.path.isdir(d)]
    if not dirs:
        return "." 
    
    def extract_num(d):
        try:
            return int(d.split('_')[-1])
        except ValueError:
            return -1
            
    return max(dirs, key=extract_num)

OUTPUT_DIR = get_target_dir()

def process_all_confusion_matrices(output_dir):
    """
    指定ディレクトリ内のすべての予測ログCSVを読み込み、
    すべてのサイクルの混同行列(HTMLとCSV)を作成・保存する
    """
    # 対象となるCSVファイルをすべて取得
    csv_files = glob.glob(os.path.join(output_dir, "detailed_predictions_log_*.csv"))
    
    if not csv_files:
        print(f"❌ '{output_dir}' 内に予測ログファイルが見つかりません。")
        return

    # 見つかったすべてのCSVファイルに対してループ処理
    for csv_filename in csv_files:
        print(f"\n📄 ファイル処理中: {os.path.basename(csv_filename)}")
        df = pd.read_csv(csv_filename)

        # カラム名の揺れに対応
        true_col = 'True_Label' if 'True_Label' in df.columns else 'True Label'
        pred_col = 'Predicted' if 'Predicted' in df.columns else 'Predicted Label'

        if 'Cycle' not in df.columns:
            print(f"⚠️ '{csv_filename}' に 'Cycle' カラムがありません。スキップします。")
            continue
            
        # CSV内に存在するすべてのサイクルを抽出してソート
        cycles = sorted(df['Cycle'].unique())
        
        # サイクルごとにループ処理
        for target_cycle in cycles:
            df_cycle = df[df['Cycle'] == target_cycle]
            if df_cycle.empty:
                continue

            y_true = df_cycle[true_col]
            y_pred = df_cycle[pred_col]

            # 混同行列の計算
            cm = confusion_matrix(y_true, y_pred, labels=range(10))
            
            cm_df = pd.DataFrame(cm, index=range(10), columns=range(10))
            cm_df.index.name = 'True \ Pred'

            mode_str = df_cycle['Mode'].iloc[0] if 'Mode' in df.columns else 'Unknown'

            # Plotly Expressのimshow(ヒートマップ)を使って描画
            fig = px.imshow(
                cm,
                text_auto=True,  # 各マスに数値を表示
                color_continuous_scale='Blues', # 青系のグラデーション
                labels=dict(x="Predicted Label (AIの予測)", y="True Label (実際の正解)", color="Count"),
                x=[str(i) for i in range(10)],
                y=[str(i) for i in range(10)],
                title=f'Confusion Matrix (Mode: {mode_str}, Cycle: {target_cycle})'
            )
            
            # レイアウトの微調整
            fig.update_layout(
                xaxis=dict(tickmode='linear', side='top'), 
                yaxis=dict(tickmode='linear'),
                width=800,
                height=800,
                template='plotly_white'
            )

            base_name = os.path.splitext(os.path.basename(csv_filename))[0]
            
            # インタラクティブなHTMLとして保存
            save_name_html = os.path.join(output_dir, f'confusion_matrix_{base_name}_cycle{target_cycle}.html')
            fig.write_html(save_name_html)
            
            # 混同行列をCSVとしても保存
            csv_save_name = os.path.join(output_dir, f'confusion_matrix_{base_name}_cycle{target_cycle}.csv')
            cm_df.to_csv(csv_save_name)
            
            print(f"  ✅ Cycle {target_cycle} のグラフとデータを保存しました！")

if __name__ == '__main__':
    # 最新の実験フォルダを自動取得して一括処理
    target_dir = get_latest_output_dir()
    print(f"📁 読み込み・保存対象フォルダ: {target_dir}")
    
    process_all_confusion_matrices(target_dir)