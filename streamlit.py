import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import os
import glob

# Streamlitのページ設定
st.set_page_config(page_title="MNIST Active Learning Dashboard", layout="wide")
st.title("MNIST Active Learning Dashboard ")

# 最新のoutputフォルダを見つける
@st.cache_data
def get_latest_output_dir():
    dirs = [d for d in glob.glob("output_*") if os.path.isdir(d)]
    if not dirs:
        return None
    return max(dirs, key=lambda d: int(d.split('_')[-1]) if d.split('_')[-1].isdigit() else -1)

latest_dir = get_latest_output_dir()

if not latest_dir:
    st.warning("実験結果 (output_X フォルダ) が見つかりません。先に main.py を実行してください。")
    st.stop()

# --- データの読み込み ---
@st.cache_data
def load_prediction_logs(output_dir):
    csv_files = glob.glob(os.path.join(output_dir, "detailed_predictions_log_*.csv"))
    if not csv_files:
        return None
    df_list = []
    for f in csv_files:
        df = pd.read_csv(f)
        df_list.append(df)
    df_all = pd.concat(df_list, ignore_index=True)

    # Confidence（予測したクラスの確率）列を作成
    if 'Confidence' not in df_all.columns:
        conf_cols = [f"Confidence_Class_{i}" for i in range(10)]
        conf_vals = df_all[conf_cols].values
        pred_idx = df_all['Predicted'].values.astype(int)
        df_all['Confidence'] = conf_vals[np.arange(len(df_all)), pred_idx]
    return df_all

df_logs = load_prediction_logs(latest_dir)

if df_logs is None:
    st.warning(f"'{latest_dir}' 内に予測ログが見つかりません。")
    st.stop()

# --- サイドバー (共通設定) ---
st.sidebar.header("設定")
modes = sorted(df_logs['Mode'].unique())
selected_mode = st.sidebar.selectbox("Mode (リセット/継続)", modes)

cycles = sorted(df_logs[df_logs['Mode'] == selected_mode]['Cycle'].unique())
selected_cycle = st.sidebar.selectbox("Cycle (サイクル数)", cycles)

# データフィルタリング
df_cycle = df_logs[(df_logs['Mode'] == selected_mode) & (df_logs['Cycle'] == selected_cycle)]

# タブで表示を切り替え
tab1, tab2, tab3 = st.tabs(["信頼度分布 (KDE)", "混同行列", "決定境界 (UMAP)"])

# ==========================================
# Tab 1: 信頼度分布 (KDE)
# ==========================================
with tab1:
    st.header("信頼度 (Confidence) の分布")
    st.markdown("モデルが予測に対してどれくらい自信を持っていたか（最大確率）をKDEプロットで可視化します。")

    # --- TP (正解データ) ---
    st.subheader("正解データ (True Positive)")
    df_tp = df_cycle[df_cycle['True Label'] == df_cycle['Predicted']]

    hist_data_tp = []
    group_labels_tp = []

    for cls in range(10):
        cls_data = df_tp[df_tp['True Label'] == cls]['Confidence'].dropna().values
        if len(cls_data) > 1: # KDEの計算には2つ以上のデータが必要
            hist_data_tp.append(cls_data)
            group_labels_tp.append(f"Class {cls}")

    if hist_data_tp:
        fig_tp = ff.create_distplot(hist_data_tp, group_labels_tp, show_hist=False, show_rug=True)
        fig_tp.update_layout(
            title="正解したデータの信頼度分布 (全クラス)",
            xaxis_title="Confidence (0.0 〜 1.0)",
            yaxis_title="Density (密度の高さ)",
            template="plotly_white",
            legend_title="True Class"
        )
        st.plotly_chart(fig_tp, use_container_width=True)
    else:
        st.info("KDEを描画するための十分な正解データがありません。")

    # --- 誤分類 (FP/FN) ---
    st.subheader("誤分類データ")
    df_wrong = df_cycle[df_cycle['True Label'] != df_cycle['Predicted']]

    if df_wrong.empty:
        st.success("誤分類されたデータはありません！")
    else:
        # ご要望に合わせ、全体表示と詳細分析を切り替えられるようにしました
        view_option = st.radio("表示モードを選択:", [
            "すべての真値クラスをまとめて比較 (概要)", 
            "特定の真値クラスを詳細に分析 (何に間違えられたか)"
        ])

        if view_option == "すべての真値クラスをまとめて比較 (概要)":
            hist_data_wrong = []
            group_labels_wrong = []
            for cls in range(10):
                cls_data = df_wrong[df_wrong['True Label'] == cls]['Confidence'].dropna().values
                if len(cls_data) > 1:
                    hist_data_wrong.append(cls_data)
                    group_labels_wrong.append(f"True Class {cls}")

            if hist_data_wrong:
                fig_wrong = ff.create_distplot(hist_data_wrong, group_labels_wrong, show_hist=False, show_rug=True)
                fig_wrong.update_layout(
                    title="誤分類されたデータの信頼度分布 (真値クラスごとに表示)",
                    xaxis_title="Confidence (0.0 〜 1.0)",
                    yaxis_title="Density (密度の高さ)",
                    template="plotly_white",
                    legend_title="True Class"
                )
                st.plotly_chart(fig_wrong, use_container_width=True)
            else:
                st.info("KDEを描画するための十分な誤分類データがありません。")

        else:
            true_classes = sorted(df_wrong['True Label'].unique())
            selected_true_class = st.selectbox("詳細を見る真のクラスを選択:", true_classes, format_func=lambda x: f"True Class {x}")

            cls_df = df_wrong[df_wrong['True Label'] == selected_true_class]
            st.write(f"**True Class {selected_true_class}** は計 {len(cls_df)} 件誤分類されました。")

            hist_data_sub = []
            group_labels_sub = []
            wrong_preds = sorted(cls_df['Predicted'].unique())

            for p_cls in wrong_preds:
                p_data = cls_df[cls_df['Predicted'] == p_cls]['Confidence'].dropna().values
                if len(p_data) > 1:
                    hist_data_sub.append(p_data)
                    group_labels_sub.append(f"Predicted as {p_cls}")

            # 1件しかデータがない場合はKDEが描けないためテキストで表示
            single_preds = [p_cls for p_cls in wrong_preds if len(cls_df[cls_df['Predicted'] == p_cls]) == 1]

            if hist_data_sub:
                fig_sub = ff.create_distplot(hist_data_sub, group_labels_sub, show_hist=False, show_rug=True)
                fig_sub.update_layout(
                    title=f"True Class {selected_true_class} が誤分類された際の信頼度 (AIが予測したクラス別)",
                    xaxis_title="Confidence (0.0 〜 1.0)",
                    yaxis_title="Density",
                    template="plotly_white",
                    legend_title="Predicted As"
                )
                st.plotly_chart(fig_sub, use_container_width=True)
            else:
                st.info("各予測クラスへの誤分類が1件以下の場合、KDEは描画されません。")

            if single_preds:
                st.warning(f"※ データが1件のみのためグラフ化されなかった予測クラス: {single_preds}")


# ==========================================
# Tab 2: 混同行列
# ==========================================
with tab2:
    st.header("混同行列 (Confusion Matrix)")
    y_true = df_cycle['True Label']
    y_pred = df_cycle['Predicted']

    cm_df = pd.crosstab(y_true, y_pred, dropna=False)
    cm_df = cm_df.reindex(index=range(10), columns=range(10), fill_value=0)

    fig_cm = px.imshow(
        cm_df.values,
        text_auto=True,
        color_continuous_scale='Blues',
        labels=dict(x="Predicted Label (AIの予測)", y="True Label (実際の正解)", color="Count"),
        x=[str(i) for i in range(10)],
        y=[str(i) for i in range(10)],
    )
    fig_cm.update_layout(
        xaxis=dict(tickmode='linear', side='top'),
        yaxis=dict(tickmode='linear'),
        width=700,
        height=700,
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# ==========================================
# Tab 3: 決定境界 (UMAP)
# ==========================================
with tab3:
    st.header("決定境界と特徴空間 (UMAP)")
    st.markdown("`visualize.py` で生成された HTML または PNG を表示します。")

    html_file = os.path.join(latest_dir, f"decision_boundary_model_weights_{selected_mode}_cycle{selected_cycle}.html")
    png_file = os.path.join(latest_dir, f"decision_boundary_model_weights_{selected_mode}_cycle{selected_cycle}.png")

    if os.path.exists(html_file):
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=850, scrolling=True)
    elif os.path.exists(png_file):
        st.image(png_file, use_container_width=True)
    else:
        st.warning(f"決定境界のグラフが見つかりません。先に visualize.py を実行してください。")