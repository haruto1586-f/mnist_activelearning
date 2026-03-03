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
from sklearn.svm import SVC
from model import get_resnet50_for_mnist  # これまで使っていたモデル定義を読み込む

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    #モデルの準備と重みのロード
    model = get_resnet50_for_mnist(device)
    weight_path = "model_weights_continue_cycle5.pt" #ロードする重みを指定
    
    match = re.search(r"model_weights_(reset|continue)_cycle(\d+)\.pt", weight_path)
    if match:
        mode_str = match.group(1)
        target_cycle = int(match.group(2))
    else:
        print("❌ エラー: ファイル名からモードとサイクルを抽出できません。")
        return
    
    try:
        #保存したデータから重み本体だけを取り出して復元
        torch.serialization.add_safe_globals([np._core.multiarray.scalar])  # NumPyの内部モジュールを安全に追加
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"モデル'{weight_path}'をロードしました．(Cycle {checkpoint.get('cycle', '?')})")
    except FileNotFoundError:
        print(f"指定された重みファイル'{weight_path}'が見つかりませんでした．")
        return
    
    #データの準備
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    np.random.seed(42)  # 再現性のためにシードを固定
    subset_indices = np.random.choice(len(test_dataset),2000, replace=False)
    test_loader = DataLoader(Subset(test_dataset, subset_indices), batch_size=256, shuffle=False)

    #特徴量の抽出
    print("特徴量を抽出中...")
    features_2048d, labels = extract_features(model, test_loader,device)
    
    #UMAPで次元削減
    print("UMAPで次元削減中...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    features_2d = reducer.fit_transform(features_2048d)
    
    #2次元空間上で代理モデル(SVM)を学習
    print("SVMで代理モデルを学習中...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm.fit(features_2d, labels)
    
    #アノテーションされたデータの抽出と配置
    print(f"Cycle {target_cycle} で新たにアノテーションされたデータを抽出しています...")
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    csv_filename = f'annotated_data_log_{mode_str}.csv'
    
    annotated_features_2d = None
    if os.path.exists(csv_filename):
        df_annotated = pd.read_csv(csv_filename)
        # 1つ前のサイクルとの差分を取り、「新しく追加されたデータ」だけを特定する
        if target_cycle > 1:
            current_indices = df_annotated[df_annotated['Cycle'] == target_cycle]['Train_Image_Index'].tolist()
            prev_indices = df_annotated[df_annotated['Cycle'] == target_cycle - 1]['Train_Image_Index'].tolist()
            new_indices = list(set(current_indices) - set(prev_indices))
        else:
            new_indices = df_annotated[df_annotated['Cycle'] == target_cycle]['Train_Image_Index'].tolist() # Cycle 1の場合は初期データ
        if new_indices:
            annotated_loader = DataLoader(Subset(train_dataset, new_indices), batch_size=256, shuffle=False)
            annotated_features_2048d, _ = extract_features(model, annotated_loader, device)
            #すでに作った2次元の地図(reducer)に当てはめる(transform)
            annotated_features_2d = reducer.transform(annotated_features_2048d)
            print(f"新たにアノテーションされた {len(new_indices)} 件のデータを配置しました！")
    else:
        print(f"ログファイル '{csv_filename}' が見つからないため、プロットをスキップします。")
    
    #メッシュグリッドの作成と境界線の予測
    x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
    y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
    
    # 0.1刻みで方眼紙のような座標点（メッシュ）を大量に作る
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # 全ての方眼紙の点に対して、SVMに「何色（0〜9）になるか」を予測させる
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    #グラフの描画
    print("グラフを描画中...")
    plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='tab10',levels=np.arange(-0.5,10.5,1), vmin=0, vmax=9)  #決定境界を背景色として淡く塗る
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, edgecolor='k', s=25, cmap='tab10', vmin=0, vmax=9, alpha=0.3)  #特徴点をクラスごとに色分けしてプロット
    
    #アノテーションされたデータを赤色の星マークで表示
    if annotated_features_2d is not None:
        plt.scatter(annotated_features_2d[:, 0], annotated_features_2d[:, 1], 
                    c='red', marker='*', s=250, edgecolors='white', linewidths=1.5, 
                    label=f'Sampled by Entropy (Cycle {target_cycle})')
        plt.legend(loc='upper right', fontsize=12)
    
    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.set_label('Digit Class (0-9)', fontsize=14)

    plt.title(f'UMAP Feature Space & Decision Boundary\n(Weights: {weight_path})', fontsize=16)
    plt.xlabel('UMAP Dimension 1', fontsize=14)
    plt.ylabel('UMAP Dimension 2', fontsize=14)
    plt.tight_layout()
    
    # 画像として保存
    base_weight_name = os.path.splitext(os.path.basename(weight_path))[0]
    save_name = f'decision_boundary_{base_weight_name}.png'
    plt.savefig(save_name)
    print(f"グラフを '{save_name}' に保存しました")
    
    plt.show()

if __name__ == '__main__':
    main()