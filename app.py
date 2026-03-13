import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import os
import glob
import sys
import streamlit.components.v1 as components

# Streamlitのページ設定
st.set_page_config(page_title="MNIST Active Learning Dashboard", layout="wide")
st.title("MNIST Active Learning Dashboard 🚀")

# 対象のoutputフォルダを見つける
@st.cache_data
def get_latest_dir():
    # Streamlitの引数(-- output_X)から取得
    for arg in sys.argv:
        if arg.startswith("output_"):
            return arg
    
    # 引数がない場合は最新を取得（保険）
    dirs = [d for d in glob.glob("output_*") if os.path.isdir(d)]
    if not dirs:
        return None
    return max(dirs, key=lambda d: int(d.split('_')[-1]) if d.split('_')[-1].isdigit() else -1)

latest_dir = get_latest_dir()

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

# 全サイクルの一覧を取得し、サイドバーでのみ選択できるように統一
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
        df_tp_plot = df_tp.copy()
        df_tp_plot['True Label'] = df_tp_plot['True Label'].astype(str)

        fig_tp = px.violin(
            df_tp_plot,
            x="True Label",
            y="Confidence",
            color="True Label",
            box=True,     
            points="all", 
            title="正解したデータの信頼度分布（X軸：正解クラス、Y軸：信頼度）",
            labels={"True Label": "正解クラス", "Confidence": "信頼度"},
            category_orders={"True Label": [str(i) for i in range(10)]}
        )
        fig_tp.update_layout(template="plotly_white", width=900, height=600)
        fig_tp.update_yaxes(range=[-0.05, 1.05]) 
        
        # === 【追加】0.5の基準線 ===
        fig_tp.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="0.5", annotation_position="top left")

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
            df_wrong_plot = df_wrong.copy()
            df_wrong_plot['True Label'] = df_wrong_plot['True Label'].astype(str)
            df_wrong_plot['Predicted'] = df_wrong_plot['Predicted'].astype(str)

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
                violinmode="group" 
            )
            fig_wrong.update_yaxes(range=[-0.05, 1.05])

            # === 【追加】0.5の基準線 ===
            fig_wrong.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="0.5", annotation_position="top left")

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
                fig_sub.update_xaxes(range=[-0.05, 1.05])
                fig_sub.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="0.5", annotation_position="top left")
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
    st.markdown("※ 予測ログ(CSV)から直接混同行列を計算して表示しています。")
    
    # 選択されているサイクルのデータから直接混同行列を計算
    y_true = df_cycle['True Label']
    y_pred = df_cycle['Predicted']

    cm_df = pd.crosstab(y_true, y_pred, dropna=False)
    # 0〜9のすべてのクラスが表に揃うように再インデックス
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
        template='plotly_white'
    )
    st.plotly_chart(fig_cm, use_container_width=True)


# ==========================================
# Tab 3: 決定境界 (UMAP)
# ==========================================
with tab3:
        st.header("3. UMAP 可視化 (学習前後 並列表示)")
        
        # 選択中のモードのログからCycleを取得
        df_annotated_temp = pd.read_csv(os.path.join(latest_dir, f"annotated_data_log_{selected_mode}.csv"))
        umap_cycles = df_annotated_temp['Cycle'].unique()
        selected_umap_cycle = st.selectbox("Cycleを選択してください", sorted(umap_cycles), key='umap_cycle')
        
        # 新しいインタラクティブHTMLのパス (サイドバーの mode 変数を使用)
        umap_html_path = os.path.join(latest_dir, f"umap_parallel_{selected_mode}_cycle{selected_umap_cycle}.html")
        
        if os.path.exists(umap_html_path):
            # HTMLファイルを読み込んで表示
            with open(umap_html_path, 'r', encoding='utf-8') as f:
                html_data = f.read()
            # 左右並列表示のため、高さを十分に確保してスクロール可能にする
            components.html(html_data, height=800, scrolling=True)
        else:
            # まだ新しいプログラムを実行していない場合のフォールバック（従来の静的画像）
            umap_png_path = os.path.join(latest_dir, f"umap_{selected_mode}_cycle{selected_umap_cycle}.png")
            if os.path.exists(umap_png_path):
                st.image(umap_png_path, use_container_width=True)
            else:
                st.warning("対象のCycleのUMAP可視化データが見つかりません。")