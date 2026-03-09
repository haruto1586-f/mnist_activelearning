import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import umap
from sklearn.neighbors import KNeighborsClassifier
import glob
from model import get_resnet50_for_mnist  # これまで使っていたモデル定義を読み込む

OUTPUT_DIR = "output"

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
    plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='tab10', levels=np.arange(-0.5, 10.5, 1), vmin=0, vmax=9)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, edgecolor='k', s=25, cmap='tab10', vmin=0, vmax=9)
    
    if annotated_features_2d is not None:
        plt.scatter(annotated_features_2d[:, 0], annotated_features_2d[:, 1],
                    facecolors='none', marker='*', s=100, edgecolors='red', linewidths=1.5, alpha=0.8,label=label_text)
        lgnd = plt.legend(loc='upper right', fontsize=12)
        if len(lgnd.legend_handles) > 0:
            lgnd.legend_handles[0].set_facecolor('red')
            lgnd.legend_handles[0].set_sizes([150])
    
    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.set_label('Digit Class (0-9)', fontsize=14)

    plt.title(f'UMAP Feature Space & Decision Boundary\n(Mode: {mode_str}, Cycle: {target_cycle})', fontsize=16)
    plt.xlabel('UMAP Dimension 1', fontsize=14)
    plt.ylabel('UMAP Dimension 2', fontsize=14)
    plt.tight_layout()
    
    # 画像として保存 (outputディレクトリ内に保存)
    base_weight_name = os.path.splitext(os.path.basename(weight_path))[0]
    save_name = os.path.join(OUTPUT_DIR, f'decision_boundary_{base_weight_name}.png')
    plt.savefig(save_name)
    print(f"グラフを '{save_name}' に保存しました")
    
    # 次のサイクルのためにメモリを解放
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"エラー: '{OUTPUT_DIR}' フォルダが存在しません。先に学習(main.py)を実行してください。")
        return

    # データの準備 (全サイクルで共通して使用)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    np.random.seed(42)  # 比較のためにテストデータの抽出シードを固定
    subset_indices = np.random.choice(len(test_dataset), 2000, replace=False)
    test_loader = DataLoader(Subset(test_dataset, subset_indices), batch_size=256, shuffle=False)

    # outputフォルダ内のすべての model_weights_*.pt ファイルを取得
    weight_files = glob.glob(os.path.join(OUTPUT_DIR, "model_weights_*.pt"))
    
    if not weight_files:
        print(f"'{OUTPUT_DIR}' フォルダに重みファイルが見つかりません。")
        return
        
    # ファイル名でソート（Cycle 1, 2, 3... の順番で処理するため）
    weight_files.sort()

    # 各重みファイルに対して順番に処理を実行
    for weight_path in weight_files:
        filename = os.path.basename(weight_path)
        # ファイル名からモード (reset/continue) と サイクル数 を抽出
        match = re.search(r"model_weights_(reset|continue)_cycle(\d+)\.pt", filename)
        
        if match:
            mode_str = match.group(1)
            target_cycle = int(match.group(2))
            process_and_plot(weight_path, mode_str, target_cycle, test_loader, train_dataset, device)
        else:
            print(f"⚠️ ファイル '{filename}' からモードとサイクルを抽出できませんでした。スキップします。")

    print("\n✅ すべてのサイクルの可視化と保存が完了しました！")

if __name__ == '__main__':
    main()