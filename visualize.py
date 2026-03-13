import os
import sys
import glob
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import umap
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from model import get_resnet50_for_mnist  

def get_target_dir():
    """コマンドライン引数があればそれを、なければ最新を取得する"""
    if len(sys.argv) > 1:
        return sys.argv[1]
    
    dirs = [d for d in glob.glob("output_*") if os.path.isdir(d)]
    if not dirs:
        return None
    
    def extract_num(d):
        try:
            return int(d.split('_')[-1])
        except ValueError:
            return -1
            
    return max(dirs, key=extract_num)

OUTPUT_DIR = get_target_dir()

def extract_features(model, dataloader, device):
    """モデルの特徴抽出部分を使って特徴量を抽出する関数"""
    model.eval()
    
    #最終の分類層を退避し「何もしない層」に置き換える
    original_fc = model.fc
    model.fc = nn.Identity()
    
    all_features = []
    all_labels = []
    all_entropies = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            features = model(inputs)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
            all_features.append(features.cpu().numpy())
            all_entropies.append(entropy.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            
    model.fc = original_fc  # 最終層を元に戻す
    return np.vstack(all_features), np.concatenate(all_labels), np.concatenate(all_entropies)

def process_and_plot(weight_path, mode_str, target_cycle, test_loader, train_dataset, device):
    """1つのサイクルの重みを読み込み、可視化して保存する処理"""
    print(f"\n--- 処理開始: Cycle {target_cycle} ({mode_str} mode) ---")
    
    # 1. モデルの準備と重みのロード
    model = get_resnet50_for_mnist(device)
    try:
        torch.serialization.add_safe_globals([np._core.multiarray.scalar])
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"モデル '{os.path.basename(weight_path)}' をロードしました。")
    except FileNotFoundError:
        print(f"指定された重みファイル '{weight_path}' が見つかりませんでした。")
        return
    
    # 2. 特徴量の抽出
    print("特徴量を抽出中...")
    features_2048d, labels, entropies = extract_features(model, test_loader, device)
    
    # 3. UMAPで次元削減
    print("UMAPで次元削減中...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    features_2d = reducer.fit_transform(features_2048d)
    
    # 4. 2次元空間上で代理モデル(k-NN)を学習
    # print("k-NNで決定境界を学習中...")
    # knn = KNeighborsClassifier(n_neighbors=5)
    # knn.fit(features_2d, labels)
    print("k-NN(回帰)で不確実性の空間分布を学習中...")
    # クラス分類用のknn_classifierを削除し、エントロピー予測用のみにする
    knn_regressor = KNeighborsRegressor(n_neighbors=5)
    knn_regressor.fit(features_2d, entropies)
    
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
            annotated_features_2048d, _, _ = extract_features(model, annotated_loader, device)
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
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    # 背景の「不確実性（エントロピー）」を予測
    Z_entropy = knn_regressor.predict(mesh_points)
    Z_entropy = Z_entropy.reshape(xx.shape)
    
    # Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)
    
    # 7. Plotlyでグラフの描画
    print("Plotlyでグラフを描画中...")
    fig = go.Figure()

    # # データ点の描画に使用する定性的カラーを取得
    # plotly_colors = px.colors.qualitative.Plotly 
    
    # # 決定境界用のカスタム定性的カラースケールを作成
    # custom_colorscale = []
    # num_classes = 10
    # for i in range(num_classes):
    #     start_norm = i / num_classes
    #     end_norm = (i + 1) / num_classes
    #     color = plotly_colors[i % len(plotly_colors)]
    #     custom_colorscale.append([start_norm, color])
    #     custom_colorscale.append([end_norm, color])

    # # 決定境界をHeatmapで描画
    # fig.add_trace(go.Heatmap(
    #     x=np.arange(x_min, x_max, 0.1),
    #     y=np.arange(y_min, y_max, 0.1),
    #     z=Z,
    #     colorscale=custom_colorscale,
    #     # ===【修正部分】背景色を薄くする ===
    #     opacity=0.35,          # 1.0 (鮮やか) から 0.35 (薄く) に変更
    #     # ================================
    #     showscale=False,      
    #     hoverinfo='skip'      
    # ))

    # # クラスごとに散布図（Scatter）を描画
    # for i in range(10):
    #     idx = labels == i
    #     fig.add_trace(go.Scatter(
    #         x=features_2d[idx, 0],
    #         y=features_2d[idx, 1],
    #         mode='markers',
    #         marker=dict(size=6, line=dict(width=0.5, color='black'), color=plotly_colors[i % len(plotly_colors)]),
    #         name=f'Class {i}',
    #         text=[f'Class {i}'] * sum(idx),
    #         hoverinfo='text+x+y'
    #     ))

    # if annotated_features_2d is not None:
    #     fig.add_trace(go.Scatter(
    #         x=annotated_features_2d[:, 0],
    #         y=annotated_features_2d[:, 1],
    #         mode='markers',
    #         marker=dict(
    #             # ===【修正部分】星マークを小さく、枠を細くする ===
    #             size=8,                   # 18 から 10 に縮小
    #             symbol='star',             
    #             color='rgba(0,0,0,0)',     
    #             line=dict(width=2, color='red') # 太さ 3 から 2 に変更
    #             # ============================================
    #         ),
    #         name=label_text,
    #         text=[label_text] * len(annotated_features_2d),
    #         hoverinfo='text+x+y'
    #     ))

    # fig.update_layout(
    #     title=f'UMAP Feature Space & Decision Boundary<br>(Mode: {mode_str}, Cycle: {target_cycle})',
    #     xaxis_title='UMAP Dimension 1',
    #     yaxis_title='UMAP Dimension 2',
    #     width=1000,
    #     height=800,
    #     legend_title='Digit Class',
    #     template='plotly_white' 
    # )
    
    # base_weight_name = os.path.splitext(os.path.basename(weight_path))[0]
    # save_name = os.path.join(OUTPUT_DIR, f'decision_boundary_{base_weight_name}.html')
    # fig.write_html(save_name)
    # print(f"インタラクティブなグラフを '{save_name}' に保存しました")
    
    # 1. 背景のHeatmap（不確実性をヒートマップで描画）
    fig.add_trace(go.Heatmap(
        x=np.arange(x_min, x_max, 0.1),
        y=np.arange(y_min, y_max, 0.1),
        z=Z_entropy,
        colorscale='Reds',      # 白（不確実性低）→ 赤（不確実性高）のグラデーション
        opacity=0.6,            # 透明度を少し下げて散布図を見やすくする
        showscale=True,         # カラーバーを表示
        colorbar=dict(title="Uncertainty<br>(Entropy)", x=1.1),
        hoverinfo='skip'      
    ))

    # データ点の描画に使用する定性的カラーを取得
    plotly_colors = px.colors.qualitative.Plotly 

    # 2. クラスごとの散布図（どの場所にどのクラスが分布しているかを重ねる）
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

    # 3. 新規アノテーションデータの強調表示
    if annotated_features_2d is not None:
        fig.add_trace(go.Scatter(
            x=annotated_features_2d[:, 0],
            y=annotated_features_2d[:, 1],
            mode='markers',
            marker=dict(
                size=10,                   
                symbol='star',             
                color='rgba(0,0,0,0)',     
                line=dict(width=2, color='white') # 背景が赤くなる可能性を考慮して白枠に変更
            ),
            name=label_text,
            text=[label_text] * len(annotated_features_2d),
            hoverinfo='text+x+y'
        ))

    fig.update_layout(
        title=f'UMAP Uncertainty (Entropy) Map<br>(Mode: {mode_str}, Cycle: {target_cycle})',
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2',
        width=1100, 
        height=800,
        legend_title='Digit Class',
        template='plotly_white' 
    )
    
    # 既存のファイル名を上書きしてしまうと分かりづらいため、名前に 'uncertainty_' を付けます
    base_weight_name = os.path.splitext(os.path.basename(weight_path))[0]
    save_name = os.path.join(OUTPUT_DIR, f'uncertainty_map_{base_weight_name}.html')
    fig.write_html(save_name)
    print(f"インタラクティブなグラフを '{save_name}' に保存しました")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if OUTPUT_DIR is None or not os.path.exists(OUTPUT_DIR):
        print(f"エラー: 出力フォルダが存在しません。先に学習(main.py)を実行してください。")
        return

    weight_files = glob.glob(os.path.join(OUTPUT_DIR, "model_weights_*.pt"))
    
    if not weight_files:
        print(f"'{OUTPUT_DIR}' フォルダに重みファイルが見つかりません。")
        return
        
    weight_files.sort()

    # すでに画像が存在するかチェック
    files_to_process = []
    for weight_path in weight_files:
        base_weight_name = os.path.splitext(os.path.basename(weight_path))[0]
        html_path = os.path.join(OUTPUT_DIR, f'decision_boundary_{base_weight_name}.html')
        png_path = os.path.join(OUTPUT_DIR, f'decision_boundary_{base_weight_name}.png')
        
        if os.path.exists(html_path) or os.path.exists(png_path):
            print(f"⏩ スキップ: 既にグラフが存在します ({base_weight_name})")
        else:
            files_to_process.append(weight_path)

    # 処理すべきファイルがない場合は終了
    if not files_to_process:
        print("\n✅ 処理すべき重みファイルがありません。")
        return

    # データの準備
    print("\nデータセットをロードしています...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    np.random.seed(42)
    subset_indices = np.random.choice(len(test_dataset), 2000, replace=False)
    test_loader = DataLoader(Subset(test_dataset, subset_indices), batch_size=256, shuffle=False)

    # 推論・描画
    for weight_path in files_to_process:
        filename = os.path.basename(weight_path)
        match = re.search(r"model_weights_(reset|continue)_cycle(\d+)\.pt", filename)
        
        if match:
            mode_str = match.group(1)
            target_cycle = int(match.group(2))
            process_and_plot(weight_path, mode_str, target_cycle, test_loader, train_dataset, device)
        else:
            print(f"⚠️ ファイル '{filename}' からモードとサイクルを抽出できませんでした。スキップします。")

    print(f"\n✅ {OUTPUT_DIR} の決定境界グラフの生成処理が完了しました！")

if __name__ == '__main__':
    main()