import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
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
    weight_path = "model_weights_reset_cycle5.pt" #ロードする重みを指定
    
    try:
        #保存したデータから重み本体だけを取り出して復元
        checkpoint = torch.load(weight_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"モデル'{weight_path}'をロードしました．(Cycle {checkpoint.get('cycle', '?')})")
    except FileNotFoundError:
        print(f"指定された重みファイル'{weight_path}'が見つかりませんでした．")
        return
    
    #データの準備
    transform = transforms.Compose([
        transforms.Greyscale(num_output_channels=3),  # ResNetは3チャンネル入力を想定しているため、グレースケールをRGBに変換
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
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='tab10')  #決定境界を背景色として淡く塗る
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, edgecolor='k', s=25, cmap='tab10')
    
    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.set_label('Digit Class (0-9)', fontsize=14)

    plt.title(f'UMAP Feature Space & Decision Boundary\n(Weights: {weight_path})', fontsize=16)
    plt.xlabel('UMAP Dimension 1', fontsize=14)
    plt.ylabel('UMAP Dimension 2', fontsize=14)
    plt.tight_layout()
    
    # 画像として保存
    save_name = 'umap_decision_boundary.png'
    plt.savefig(save_name)
    print(f"グラフを '{save_name}' に保存しました")
    
    plt.show()

if __name__ == '__main__':
    main()