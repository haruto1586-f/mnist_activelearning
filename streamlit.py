# app.py
import streamlit as st
import os
import glob
from PIL import Image

# Streamlitのページ設定
st.set_page_config(page_title="MNIST Active Learning Dashboard", layout="wide")

st.title("MNIST Active Learning Visualization 🚀")
st.markdown("UMAPによる特徴空間と決定境界、および能動学習でサンプリングされたデータの可視化")

# カレントディレクトリにある "decision_boundary_*.png" の画像一覧を取得
image_files = glob.glob("decision_boundary_*.png")
image_files.sort() # 名前順にソート

if not image_files:
    st.warning("表示できる画像がありません。先に visualize.py を実行して画像を生成してください。")
else:
    # サイドバーに設定用のUIを配置
    st.sidebar.header("設定")
    
    # セレクトボックスで表示したい画像を選択できるようにする
    selected_image_path = st.sidebar.selectbox(
        "表示するグラフを選択してください:",
        image_files
    )

    # 選択された画像を読み込んで表示
    if selected_image_path:
        image = Image.open(selected_image_path)
        st.image(image, caption=f"表示中: {selected_image_path}", use_container_width=True)

        # 画像ファイル名からモードとサイクルを推測して表示（任意）
        import re
        match = re.search(r"model_weights_(reset|continue)_cycle(\d+)", selected_image_path)
        if match:
            mode = match.group(1)
            cycle = match.group(2)
            st.sidebar.success(f"現在の表示:\n- モード: {mode}\n- サイクル: {cycle}")