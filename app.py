import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import os
import glob
import sys

# Streamlitのページ設定
st.set_page_config(page_title="MNIST Active Learning Dashboard", layout="wide")
st.title("MNIST Active Learning Dashboard 🚀")

# 対象のoutputフォルダを見つける
@st.cache_data
def get_target_dir():
    # Streamlitの引数(-- output_X)から取得
    for arg in sys.argv:
        if arg.startswith("output_"):
            return arg
    
    # 引数がない場合は最新を取得（保険）
    dirs = [d for d in glob.glob("output_*") if os.path.isdir(d)]
    if not dirs:
        return None
    return max(dirs, key=lambda d: int(d.split('_')[-1]) if d.split('_')[-1].isdigit() else -1)

latest_dir = get_target_dir()

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
st.sidebar.header("⚙️ 共通設定")
modes = sorted(df_logs['Mode'].unique())
selected_mode = st.sidebar.selectbox("Mode (リセット/継続)", modes)

# 全サイクルの一覧を取得し、サイドバーで選択できるようにする
cycles = sorted(df_logs[df_logs['Mode'] == selected_mode]['Cycle'].unique())
selected_cycle = st.sidebar.selectbox("Cycle (サイクル数)", cycles)

# 選択されたモードとサイクルでデータをフィルタリング（全タブ共通）
df_cycle = df_logs[(df_logs['Mode'] == selected_mode) & (df_logs['Cycle'] == selected_cycle)]


# タブの作成
tab1, tab2, tab3 = st.tabs(["📊 信頼度分布 (KDE/バイオリン)", "🔥 混同行列", "🗺️ 決定境界 (UMAP)"])

# ==========================================
# Tab 1: 信頼度分布 (バイオリンプロット / KDE)
# ==========================================
with tab1:
    st.header(f"信頼度 (Confidence) の分布 - Cycle {selected_cycle}")

    # --- TP (正解データ) ---
    st.subheader("✅ 正解データ (True Positive)")
    df_tp = df_cycle[df_cycle['True Label'] == df_cycle['Predicted']]

    if df_tp.empty:
        st.info("正解データがありません。")
    else:
        # 色分けのためにクラスを文字列型に変換
        df_tp_plot = df_tp.copy()
        df_tp_plot['True Label'] = df_tp_plot['True Label'].astype(str)

        # バイオリンプロット（箱ひげ図とすべてのデータ点を併記）
        fig_tp = px.violin(
            df_tp_plot,
            x="True Label",
            y="Confidence",
            color="True Label",
            box=True,     # 箱ひげ図を表示
            points="all", # すべてのデータ点を散らして表示
            title="正解したデータの信頼度分布（X軸：正解クラス、Y軸：信頼度）",
            labels={"True Label": "正解クラス", "Confidence": "信頼度"},
            category_orders={"True Label": [str(i) for i in range(10)]}
        )
        fig_tp.update_layout(template="plotly_white", width=900, height=600)
        # 信頼度の軸を0〜1（+少しの余白）に固定
        fig_tp.update_yaxes(range=[-0.05, 1.05]) 
        st.plotly_chart(fig_tp, use_container_width=True)


    # --- FP/FN (誤分類データ) ---
    st.subheader("❌ 誤分類データ")
    df_wrong = df_cycle[df_cycle['True Label'] != df_cycle['Predicted']]

    if df_wrong.empty:
        st.success("誤分類されたデータはありません！🎉")
    else:
        view_option = st.radio("表示モードを選択:", [
            "すべての真値クラスをまとめて比較 (バイオリンプロット一覧)", 
            "特定の真値クラスを詳細に分析 (KDE分布)"
        ])

        if view_option == "すべての真値クラスをまとめて比較 (バイオリンプロット一覧)":
            # 色をカテゴリとして扱うために文字列に変換
            df_wrong_plot = df_wrong.copy()
            df_wrong_plot['True Label'] = df_wrong_plot['True Label'].astype(str)
            df_wrong_plot['Predicted'] = df_wrong_plot['Predicted'].astype(str)

            # バイオリンプロットに変更
            fig_wrong = px.violin(
                df_wrong_plot, 
                x="True Label", 
                y="Confidence", 
                color="Predicted",
                box=True,
                points="all",
                title="誤分類データの信頼度分布（X軸：実際の正解、Y軸：信頼度、色：AIの誤予測クラス）",
                labels={"True Label": "実際の正解 (True Class)", "Predicted": "AIの誤予測 (Predicted)", "Confidence": "信頼度"},
                category_orders={
                    "True Label": [str(i) for i in range(10)],
                    "Predicted": [str(i) for i in range(10)]
                }
            )
            fig_wrong.update_layout(
                template="plotly_white", 
                width=1000,  
                height=600,
                violinmode="group" # 予測クラスごとにバイオリンを横に並べる
            )
            # 信頼度の軸を0〜1に固定
            fig_wrong.update_yaxes(range=[-0.05, 1.05])
            st.plotly_chart(fig_wrong, use_container_width=True)

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

            single_preds = [p_cls for p_cls in wrong_preds if len(cls_df[cls_df['Predicted'] == p_cls]) == 1]

            if hist_data_sub:
                fig_sub = ff.create_distplot(hist_data_sub, group_labels_sub, show_hist=False, show_rug=True)
                fig_sub.update_layout(
                    title=f"True Class {selected_true_class} が誤分類された際の信頼度分布 (AIの予測クラス別)",
                    xaxis_title="Confidence (0.0 〜 1.0)",
                    yaxis_title="Density",
                    template="plotly_white",
                    legend_title="Predicted As"
                )
                # X軸（KDEグラフでの信頼度軸）を0〜1に固定
                fig_sub.update_xaxes(range=[-0.05, 1.05])
                st.plotly_chart(fig_sub, use_container_width=True)
            else:
                st.info("各予測クラスへの誤分類が1件以下の場合、KDEは描画されません。")

            if single_preds:
                st.warning(f"※ データが1件のみのためKDE化されなかった予測クラス: {single_preds}")


# ==========================================
# Tab 2: 混同行列
# ==========================================
with tab2:
    st.header(f"混同行列 (Confusion Matrix) - Cycle {selected_cycle}")
    st.markdown("`confusion_matrix.py` で生成された HTML を表示します。")
    
    html_file_cm = os.path.join(latest_dir, f"confusion_matrix_detailed_predictions_log_{selected_mode}_cycle{selected_cycle}.html")

    if os.path.exists(html_file_cm):
        with open(html_file_cm, 'r', encoding='utf-8') as f:
            html_content_cm = f.read()
        st.components.v1.html(html_content_cm, height=850, scrolling=True)
    else:
        st.warning(f"Cycle {selected_cycle} の混同行列グラフが見つかりません。先に confusion_matrix.py を実行してください。")


# ==========================================
# Tab 3: 決定境界 (UMAP)
# ==========================================
with tab3:
    st.header(f"決定境界と特徴空間 (UMAP) - Cycle {selected_cycle}")
    st.markdown("`visualize.py` で生成された HTML または PNG を表示します。")

    html_file_umap = os.path.join(latest_dir, f"decision_boundary_model_weights_{selected_mode}_cycle{selected_cycle}.html")
    png_file_umap = os.path.join(latest_dir, f"decision_boundary_model_weights_{selected_mode}_cycle{selected_cycle}.png")

    if os.path.exists(html_file_umap):
        with open(html_file_umap, 'r', encoding='utf-8') as f:
            html_content_umap = f.read()
        st.components.v1.html(html_content_umap, height=850, scrolling=True)
    elif os.path.exists(png_file_umap):
        st.image(png_file_umap, use_container_width=True)
    else:
        st.warning(f"Cycle {selected_cycle} の決定境界グラフが見つかりません。先に visualize.py を実行してください。")