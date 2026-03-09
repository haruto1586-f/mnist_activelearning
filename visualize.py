import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import re
import plotly.graph_objects as go
import plotly.express as px
import umap
from sklearn.neighbors import KNeighborsClassifier
import glob
from model import get_resnet50_for_mnist  # これまで使っていたモデル定義を読み込む

def get_latest_output_dir():
    """最新の output_X フォルダを取得する"""
    dirs = [d for d in glob.glob("output_*") if os.path.isdir(d)]
    if not dirs:
        return None
    
    # フォルダ名末尾の数字部分でソートして一番大きいものを返す
    def extract_num(d):
        try:
            return int(d.split('_')[-1])
        except ValueError:
            return -1
            
    latest_dir = max(dirs, key=extract_num)
    return latest_dir

# 最新のフォルダを自動取得
OUTPUT_DIR = get_latest_output_dir()

def extract_features(model, dataloader, device):
    """モデルの特徴抽出部分を使って特徴量を抽出する関数"""
    model.eval()
    
    #最終の分類層を退避し「何もしない層」に置き換える
    original_fc = model.fc
    model.fc = nn.Identity()
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            features = model(inputs)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    model.fc = original_fc  # 最終層を元に戻す
    return np.vstack(all_features), np.concatenate(all_labels)

def process_and_plot(weight_path, mode_str, target_cycle, test_loader, train_dataset, device):
    """1つのサイクルの重みを読み込み、可視化して保存する処理"""
    print(f"\n--- 処理開始: Cycle {target_cycle} ({mode_str} mode) ---")
    
    # 1. モデルの準備と重みのロード
    model = get_resnet50_for_mnist(device)
    try:
        torch.serialization.add_safe_globals([np._core.multiarray.scalar])
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"モデル '{weight_path}' をロードしました。")
    except FileNotFoundError:
        print(f"指定された重みファイル '{weight_path}' が見つかりませんでした。")
        return
    
    # 2. 特徴量の抽出
    print("特徴量を抽出中...")
    features_2048d, labels = extract_features(model, test_loader, device)
    
    # 3. UMAPで次元削減
    print("UMAPで次元削減中...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    features_2d = reducer.fit_transform(features_2048d)
    
    # 4. 2次元空間上で代理モデル(k-NN)を学習
    print("k-NNで決定境界を学習中...")
    knn = KNeighborsClassifier(n_neighbors=5)  # k=5のk近傍法を使用
    knn.fit(features_2d, labels)
    
    # 5. アノテーションされたデータの抽出と配置
    csv_filename = os.path.join(OUTPUT_DIR, f'annotated_data_log_{mode_str}.csv')
    annotated_features_2d = None
    if os.path.exists(csv_filename):
        df_annotated = pd.read_csv(csv_filename)
        # 1つ前のサイクルとの差分を取り、「新しく追加されたデータ」だけを特定する
        if target_cycle > 1:
            current_indices = df_annotated[df_annotated['Cycle'] == target_cycle]['Train_Image_Index'].tolist()
            prev_indices = df_annotated[df_annotated['Cycle'] == target_cycle - 1]['Train_Image_Index'].tolist()
            new_indices = list(set(current_indices) - set(prev_indices))
            label_text = f'Sampled by Active Learning (Cycle {target_cycle})'
        else:
            new_indices = df_annotated[df_annotated['Cycle'] == target_cycle]['Train_Image_Index'].tolist()
            label_text = 'Initial Random Samples'
            
        if new_indices:
            annotated_loader = DataLoader(Subset(train_dataset, new_indices), batch_size=256, shuffle=False)
            annotated_features_2048d, _ = extract_features(model, annotated_loader, device)
            # すでに作った2次元の地図(reducer)に当てはめる
            annotated_features_2d = reducer.transform(annotated_features_2048d)
            print(f"新たにアノテーションされた {len(new_indices)} 件のデータを配置しました！")
    else:
        print(f"ログファイル '{csv_filename}' が見つからないため、プロットをスキップします。")
    
    # 6. メッシュグリッドの作成と境界線の予測
    x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
    y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 7. グラフの描画
    print("グラフを描画中...")
    fig = go.Figure()
    
    # 決定境界をContour（等高線）で描画
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, 0.1),
        y=np.arange(y_min, y_max, 0.1),
        z=Z,
        colorscale='Plotly3', # Plotly標準のカラースケール
        opacity=0.3,          # 背景として薄く表示
        showscale=True,      # カラーバーは表示
        hoverinfo='skip'      # ホバー時の情報は表示しない
    ))

    # クラスごとに散布図（Scatter）を描画
    plotly_colors = px.colors.qualitative.Plotly
    for i in range(10):
        idx = labels == i
        fig.add_trace(go.Scatter(
            x=features_2d[idx, 0],
            y=features_2d[idx, 1],
            mode='markers',
            marker=dict(size=6, line=dict(width=0.5, color='black'), color=plotly_colors[i % len(plotly_colors)]),
            name=f'Class {i}',
            text=[f'Class {i}'] * sum(idx),
            hoverinfo='text+x+y'
        ))

    # アノテーションされたデータを赤色の星マークで追加
    if annotated_features_2d is not None:
        fig.add_trace(go.Scatter(
            x=annotated_features_2d[:, 0],
            y=annotated_features_2d[:, 1],
            mode='markers',
            marker=dict(
                size=18,           # 少し大きめに設定
                symbol='star',             # 星型
                color='rgba(0,0,0,0)',     # 中身を透明にする
                line=dict(width=3, color='red') # 赤くて太い枠線
            ),
            name=label_text,
            text=[label_text] * len(annotated_features_2d),
            hoverinfo='text+x+y'
        ))

    # レイアウトの調整
    fig.update_layout(
        title=f'UMAP Feature Space & Decision Boundary<br>(Mode: {mode_str}, Cycle: {target_cycle})',
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2',
        width=1000,
        height=800,
        legend_title='Digit Class',
        template='plotly_white' # 背景を白にする
    )
    
    # 画像（HTML）として保存
    base_weight_name = os.path.splitext(os.path.basename(weight_path))[0]
    save_name = os.path.join(OUTPUT_DIR, f'decision_boundary_{base_weight_name}.html')
    fig.write_html(save_name)
    print(f"インタラクティブなグラフを '{save_name}' に保存しました")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"エラー: '{OUTPUT_DIR}' フォルダが存在しません。先に学習(main.py)を実行してください。")
        return

    # データの準備
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    np.random.seed(42)
    subset_indices = np.random.choice(len(test_dataset), 2000, replace=False)
    test_loader = DataLoader(Subset(test_dataset, subset_indices), batch_size=256, shuffle=False)

    # outputフォルダ内のすべての model_weights_*.pt ファイルを取得
    weight_files = glob.glob(os.path.join(OUTPUT_DIR, "model_weights_*.pt"))
    
    if not weight_files:
        print(f"'{OUTPUT_DIR}' フォルダに重みファイルが見つかりません。")
        return
        
    weight_files.sort()

    for weight_path in weight_files:
        filename = os.path.basename(weight_path)
        match = re.search(r"model_weights_(reset|continue)_cycle(\d+)\.pt", filename)
        
        if match:
            mode_str = match.group(1)
            target_cycle = int(match.group(2))
            process_and_plot(weight_path, mode_str, target_cycle, test_loader, train_dataset, device)
        else:
            print(f" ファイル '{filename}' からモードとサイクルを抽出できませんでした。スキップします。")

    print("\n すべてのサイクルの可視化と保存が完了しました！")

if __name__ == '__main__':
    main()