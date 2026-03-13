import os
import glob
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import umap
from model import get_resnet50_for_mnist  

def get_target_dir():
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
    if len(dataloader.dataset) == 0:
        return np.array([]), np.array([])
        
    model.eval()
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
            
    model.fc = original_fc
    return np.vstack(all_features), np.concatenate(all_labels)

def get_dataloader_for_indices(dataset, indices, batch_size=256):
    if len(indices) == 0:
        return DataLoader(Subset(dataset, []), batch_size=batch_size)
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)

def build_umap_trace(features_2d, labels, name_prefix, marker_symbol, opacity, size, line_width=0, line_color=None):
    """PlotlyのScatterトレースを生成するヘルパー関数"""
    traces = []
    plotly_colors = px.colors.qualitative.Plotly
    
    if len(features_2d) == 0:
        return traces
        
    for i in range(10):
        idx = labels == i
        if sum(idx) == 0:
            continue
            
        marker_dict = dict(
            size=size,
            symbol=marker_symbol,
            color=plotly_colors[i % len(plotly_colors)],
            opacity=opacity
        )
        if line_width > 0:
            marker_dict['line'] = dict(width=line_width, color=line_color if line_color else 'black')
            
        traces.append(go.Scatter(
            x=features_2d[idx, 0],
            y=features_2d[idx, 1],
            mode='markers',
            marker=marker_dict,
            name=f'{name_prefix} (Class {i})',
            legendgroup=f'class_{i}',  # 共通の凡例グループに紐付け
            showlegend=False,          # 個別のトレースは凡例に表示しない
            text=[f'[{name_prefix}] Class {i}'] * sum(idx), # ホバー時にデータ種別とクラスを表示
            hoverinfo='text'
        ))
    return traces

def process_cycle_parallel(cycle, mode_str, train_dataset, test_dataset, df_annotated, device, sample_size=2000):
    print(f"\n--- 並列表示処理開始: Cycle {cycle} ({mode_str} mode) ---")
    
    prev_cycle = cycle - 1
    weight_path_prev = os.path.join(OUTPUT_DIR, f"model_weights_{mode_str}_cycle{prev_cycle}.pt")
    weight_path_curr = os.path.join(OUTPUT_DIR, f"model_weights_{mode_str}_cycle{cycle}.pt")
    
    if not os.path.exists(weight_path_curr):
        print(f"Cycle {cycle} の重みが見つかりません。")
        return
    if not os.path.exists(weight_path_prev):
        print(f"Cycle {prev_cycle} の重みが見つかりません。スキップします。")
        return

    current_indices = df_annotated[df_annotated['Cycle'] == cycle]['Train_Image_Index'].tolist()
    if prev_cycle >= 1:
        prev_indices = df_annotated[df_annotated['Cycle'] == prev_cycle]['Train_Image_Index'].tolist()
    else:
        prev_indices = []
        if cycle == 1:
            prev_indices = current_indices.copy()
    
    if cycle == 1:
        new_indices = current_indices
        old_indices = []
        new_label_text = "Initial Random"
    else:
        new_indices = list(set(current_indices) - set(prev_indices))
        old_indices = prev_indices
        new_label_text = "Newly Annotated"

    all_train_indices = set(range(len(train_dataset)))
    unlabeled_pool = list(all_train_indices - set(current_indices))
    
    np.random.seed(42)
    unlabeled_sample_indices = np.random.choice(unlabeled_pool, min(sample_size, len(unlabeled_pool)), replace=False).tolist()
    test_sample_indices = np.random.choice(len(test_dataset), min(sample_size, len(test_dataset)), replace=False).tolist()

    dl_old = get_dataloader_for_indices(train_dataset, old_indices)
    dl_new = get_dataloader_for_indices(train_dataset, new_indices)
    dl_curr_all = get_dataloader_for_indices(train_dataset, current_indices)
    dl_unlabeled = get_dataloader_for_indices(train_dataset, unlabeled_sample_indices)
    dl_test = get_dataloader_for_indices(test_dataset, test_sample_indices)

    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=(f"Before Learning (Model: Cycle {prev_cycle})", 
                                        f"After Learning (Model: Cycle {cycle})"))

    plotly_colors = px.colors.qualitative.Plotly

    # === 凡例用のダミートレースを追加（Class 0〜9） ===
    # グラフには描画されない透明な点を追加し、これのみを凡例として表示させます
    for i in range(10):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=12, color=plotly_colors[i % len(plotly_colors)], symbol='circle'),
            name=f'Class {i}',
            legendgroup=f'class_{i}',
            showlegend=True
        ), row=1, col=1)

    models_info = [
        (weight_path_prev, 1), 
        (weight_path_curr, 2)
    ]

    for weight_path, col_idx in models_info:
        model = get_resnet50_for_mnist(device)
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Col {col_idx}: 特徴量抽出中...")
        feat_old, labels_old = extract_features(model, dl_old, device)
        feat_new, labels_new = extract_features(model, dl_new, device)
        feat_unlabeled, labels_unlabeled = extract_features(model, dl_unlabeled, device)
        feat_test, labels_test = extract_features(model, dl_test, device)
        feat_curr_all, labels_curr_all = extract_features(model, dl_curr_all, device)

        print(f"Col {col_idx}: UMAPマッピング中...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        
        if col_idx == 1:
            fit_features = feat_old if len(feat_old) > 0 else feat_curr_all
            reducer.fit(fit_features)
        else:
            reducer.fit(feat_curr_all)

        umap_old = reducer.transform(feat_old) if len(feat_old) > 0 else np.array([])
        umap_new = reducer.transform(feat_new) if len(feat_new) > 0 else np.array([])
        umap_unlabeled = reducer.transform(feat_unlabeled)
        umap_test = reducer.transform(feat_test)

        # トレースの追加 (showlegend=Falseは関数内で処理済み)
        traces_unlabeled = build_umap_trace(umap_unlabeled, labels_unlabeled, 'Unlabeled', 'circle', opacity=0.15, size=4)
        traces_test = build_umap_trace(umap_test, labels_test, 'Test', 'cross', opacity=0.15, size=4)
        for t in traces_unlabeled + traces_test:
            fig.add_trace(t, row=1, col=col_idx)

        traces_old = build_umap_trace(umap_old, labels_old, 'Labeled (Old)', 'circle', opacity=1.0, size=7, line_width=1)
        for t in traces_old:
            fig.add_trace(t, row=1, col=col_idx)

        traces_new = build_umap_trace(umap_new, labels_new, new_label_text, 'star', opacity=1.0, size=14, line_width=1, line_color='red')
        for t in traces_new:
            fig.add_trace(t, row=1, col=col_idx)

    # レイアウトの調整
    fig.update_layout(
        title_text=f'UMAP Active Learning Transition (Mode: {mode_str}, Cycle: {cycle})<br>'
                   f'<span style="font-size:13px;">Left: Model before learning | Right: Model after learning (Fit on Labeled, Transform Unlabeled/Test)</span><br>'
                   f'<span style="font-size:13px;"><b>[Markers]  ★: Newly Annotated | ●: Labeled (Old) | ◯(faded): Unlabeled | ＋(faded): Test</b></span>',
        height=750,
        width=1400,
        template='plotly_white',
        showlegend=True,
        legend_title_text="Class Filter"
    )
    
    fig.update_xaxes(title_text="UMAP 1", row=1, col=1)
    fig.update_yaxes(title_text="UMAP 2", row=1, col=1)
    fig.update_xaxes(title_text="UMAP 1", row=1, col=2)
    fig.update_yaxes(title_text="UMAP 2", row=1, col=2)

    save_name = os.path.join(OUTPUT_DIR, f'umap_parallel_{mode_str}_cycle{cycle}.html')
    fig.write_html(save_name)
    print(f"✅ グラフを保存しました: {save_name}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if OUTPUT_DIR is None or not os.path.exists(OUTPUT_DIR):
        print("エラー: 出力フォルダが存在しません。先に main.py を実行してください。")
        return

    print("\nデータセットをロードしています...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    csv_files = glob.glob(os.path.join(OUTPUT_DIR, "annotated_data_log_*.csv"))
    
    for csv_file in csv_files:
        mode_match = re.search(r"annotated_data_log_(reset|continue)\.csv", os.path.basename(csv_file))
        if not mode_match:
            continue
        
        mode_str = mode_match.group(1)
        df_annotated = pd.read_csv(csv_file)
        cycles = df_annotated['Cycle'].unique()

        for cycle in sorted(cycles):
            process_cycle_parallel(cycle, mode_str, train_dataset, test_dataset, df_annotated, device)

    print(f"\n🎉 全ての並列可視化処理が完了しました！")

if __name__ == '__main__':
    main()