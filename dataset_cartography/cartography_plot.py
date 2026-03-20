import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os

# ページ全体の設定 (横幅を広く使う)
st.set_page_config(page_title="Cartography Dashboard", layout="wide")

st.title("Dataset Cartography Analysis Dashboard")

# CSVデータの読み込み (キャッシュ化して再読み込みを高速化)
@st.cache_data
def load_data(csv_path):
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)

output_dir = "cartography_output"
csv_path = os.path.join(output_dir, "dataset_cartography_metrics.csv")

st.sidebar.header("Settings")
dataset_choice = st.sidebar.radio("Select Dataset:", ["Train Data", "Test Data"])

# 選ばれたデータセットに応じて読み込むCSVを変更
if dataset_choice == "Train Data":
    csv_path = os.path.join(output_dir, "train_cartography_metrics.csv")
else:
    csv_path = os.path.join(output_dir, "test_cartography_metrics.csv")

df = load_data(csv_path)

if df is None:
    st.warning(f"データが見つかりません: `{csv_path}`\n先に main.py を実行してCSVファイルを生成してください。")
else:
    max_epochs = df['Epoch'].max()

    # サイドバーにサンプリング数を調整できるスライダーを追加！
    st.sidebar.header("Settings")
    #エポック指定のスライダー
    current_epoch = st.sidebar.slider(
        "Select Epoch up to:", 
        min_value=1, 
        max_value=max_epochs, 
        value=max_epochs, 
        step=1
    )
    sample_size = st.sidebar.slider("Sampling size for line charts", min_value=5, max_value=50, value=15, step=5)

    # ==========================================
    # 1. Data Map (散布図)
    # ==========================================
    # スライダーで指定されたエポック「以前」のデータのみを抽出
    df_filtered = df[df['Epoch'] <= current_epoch].copy()
    df_filtered['Prob'] = np.exp(-df_filtered['Loss'])

    df_metrics = df_filtered.groupby('Index').agg(
        Dynamic_Confidence=('Prob', 'mean'),
        Dynamic_Variability=('Prob', 'std'),
        True_Label=('True_Label', 'first')
    ).reset_index()

    df_metrics['Dynamic_Variability'] = df_metrics['Dynamic_Variability'].fillna(0)
    
    df_current = df[df['Epoch'] == current_epoch].copy()
    df_metrics = pd.merge(df_metrics, df_current[['Index', 'Predicted_Label']], on='Index')
    
    df_metrics['True_Label'] = df_metrics['True_Label'].astype(str)
    
    # 3つの領域に分類
    df_metrics['True_Label'] = df_metrics['True_Label'].astype(str)
    df_metrics = df_metrics.sort_values(by='True_Label')
    
    conditions = [
        (df_metrics['Dynamic_Confidence'] >= 0.7) & (df_metrics['Dynamic_Variability'] <= 0.25),
        (df_metrics['Dynamic_Confidence'] <= 0.3) & (df_metrics['Dynamic_Variability'] <= 0.25)
    ]
    choices = ['Easy-to-Learn', 'Hard-to-Learn']
    df_metrics['Region'] = np.select(conditions, choices, default='Ambiguous')

    # 凡例の並び順を綺麗にするためにソート
    df_metrics['Region'] = pd.Categorical(df_metrics['Region'], categories=['Easy-to-Learn', 'Ambiguous', 'Hard-to-Learn'], ordered=True)
    df_metrics = df_metrics.sort_values('Region')
    
    #データマップの描画
    st.subheader(f"Dataset Cartography Map (Up to Epoch {current_epoch})")
    
    color_map = {
        'Easy-to-Learn': '#2ca02c', # 緑色
        'Ambiguous': '#ff7f0e',     # オレンジ色
        'Hard-to-Learn': '#d62728'  # 赤色
    }

    # 色分けを Region に固定し、凡例を追加
    fig_map = px.scatter(
        df_metrics, x='Dynamic_Variability', y='Dynamic_Confidence', color='Region',
        hover_data=['Index', 'True_Label', 'Predicted_Label'], 
        color_discrete_map=color_map,
        labels={
            'Dynamic_Variability': 'Variability (Standard Deviation)',
            'Dynamic_Confidence': 'Confidence (Mean)',
            'Region': 'Cartography Region' # 凡例のタイトル
        },
        opacity=0.6,
        range_x=[-0.02, 0.45], 
        range_y=[-0.05, 1.05]
    )
    fig_map.update_traces(marker=dict(size=4))
    
    # 凡例のデザイン調整（クリック操作がしやすいように少し大きく配置）
    fig_map.update_layout(
        legend=dict(
            title="Cartography Region",
            itemsizing='constant',
            font=dict(size=14)
        )
    )
    
    st.plotly_chart(fig_map, use_container_width=True)

    # ==========================================
    # 3. 折れ線グラフのデータ準備 (現在のエポック基準でTP/FPを判定)
    # ==========================================
    df_current['Prediction_Type'] = np.where(
        df_current['Predicted_Label'] == df_current['True_Label'], 
        'TP (Correct)', 
        'FP (Incorrect)'
    )
    
    tp_indices = df_current[df_current['Prediction_Type'] == 'TP (Correct)']['Index'].sample(n=sample_size, random_state=42, replace=True).unique()
    fp_indices = df_current[df_current['Prediction_Type'] == 'FP (Incorrect)']['Index'].sample(n=sample_size, random_state=42, replace=True).unique()
    
    def prep_trend_df(indices):
        # 折れ線グラフは「指定エポックまでの推移」を見せる
        return df_filtered[df_filtered['Index'].isin(indices)].copy()

    df_tp = prep_trend_df(tp_indices)
    df_fp = prep_trend_df(fp_indices)

    # ==========================================
    # 4 & 5. Confidence Trend (横に2つ並べる)
    # ==========================================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"TP (Correct at Epoch {current_epoch})")
        fig_tp = px.line(
            df_tp, x='Epoch', y='Prob', line_group='Index',
            hover_data=['Index', 'True_Label', 'Predicted_Label'],
            labels={'Prob': 'Gold Label Probability', 'Epoch': 'Epoch'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_tp.update_traces(mode='lines+markers', marker=dict(size=6), opacity=0.7)
        # X軸も最大エポック数で固定して、線が伸びていくアニメーション効果を出す
        fig_tp.update_layout(yaxis_range=[-0.05, 1.05], xaxis_range=[0.5, max_epochs + 0.5])
        fig_tp.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="0.5 Threshold", annotation_position="top left")   
        st.plotly_chart(fig_tp, use_container_width=True)

    with col2:
        st.subheader(f"FP (Incorrect at Epoch {current_epoch})")
        fig_fp = px.line(
            df_fp, x='Epoch', y='Prob', line_group='Index',
            hover_data=['Index', 'True_Label', 'Predicted_Label'],
            labels={'Prob': 'Gold Label Probability', 'Epoch': 'Epoch'},
            color_discrete_sequence=['#ff7f0e']
        )
        fig_fp.update_traces(mode='lines+markers', marker=dict(size=6), opacity=0.7)
        fig_fp.update_layout(yaxis_range=[-0.05, 1.05], xaxis_range=[0.5, max_epochs + 0.5])
        fig_fp.add.hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="0.5 Threshold", annotation_position="top left")
        st.plotly_chart(fig_fp, use_container_width=True)