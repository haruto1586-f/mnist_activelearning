import sys
import os
import subprocess

def main():
    if len(sys.argv) < 2:
        print("==================================================")
        print("❌ エラー: 参照するフォルダ名を指定してください。")
        print("使用例: python run_dashboard.py output_1")
        print("==================================================")
        sys.exit(1)

    target_dir = sys.argv[1]

    if not os.path.exists(target_dir):
        print(f"❌ エラー: フォルダ '{target_dir}' が見つかりません。")
        sys.exit(1)

    print(f"🚀 '{target_dir}' の可視化ダッシュボードを準備します...\n")

    # 1. 決定境界の生成
    print(f"▶️ [1/2] 決定境界グラフを作成中...")
    subprocess.run([sys.executable, "visualize.py", target_dir])

    # 2. Streamlitの起動
    print(f"\n▶️ [2/2] ダッシュボードを起動します...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--", target_dir])

if __name__ == "__main__":
    main()