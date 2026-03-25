import time
import random

import wandb


def main():

    # プロジェクトの初期化
    # 実行すると、クラウド上に「my-first-project」が自動で作られます
    wandb.init(
        entity="charmie11-tottori-university",  # Team名
        project="mnist-activelearning-wandb-1st-step"
    )

    # 模擬的な学習ループ
    for i in range(20):
        loss = 1.0 / (i + 1) + random.random() * 0.1
        acc = 1.0 - loss

        # クラウドにデータを送信
        wandb.log({"loss": loss, "accuracy": acc, "step": i})

        print(f"Step {i}: loss={loss:.4f}")
        time.sleep(0.5) # グラフが動くのを見やすくするため

    # 終了処理
    wandb.finish()


if __name__ == "__main__":
    main()
