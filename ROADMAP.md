# Playground v2.5 学習のロードマップ

## 最終目標

Playground v2.5（`playgroundai/playground-v2.5-1024px-aesthetic`）で、SDXL 用の LoRA を正しい EDM の式で学習する。そのあとフルファインチューンも同じ式で学習する。想定は 16GB GPU、画像は最大およそ 40 万枚。できた LoRA は ComfyUI で公式チェックポイントの上に載せて使う。

## 段階

| 段階 | 状態 | 目安 |
|---|---|---|
| 公式の式・キャッシュの分離・危険な DDPM オプションの拒否を調べる | 終わった | 100% |
| SDXL LoRA（`sdxl_train_network.py`）に EDM を入れる | 終わった。CPU のテストと小さいモデルの学習は通っている | 100% |
| レビューで指摘された周辺（検証の σ、キャッシュの壊れ、σ の分布、損失の重み、重み比較スクリプト） | コードと CPU テストは入れた。実 GPU では未確認 | 70% |
| 公式の重みで kohya と diffusers の UNet が一致するか | スクリプトだけある。まだ実行していない | 0% |
| 16GB の Windows PC で短い学習、VRAM と速度の確認 | やっていない | 0% |
| σ の分布と損失の重みを実 GPU で比べ、既定値を決める | やっていない。今の既定は diffusers と同じ（一様 Karras、重みなし） | 0% |
| ComfyUI で LoRA を読み、絵が崩れていないか見る | やっていない | 0% |
| GUI のプリセット | やっていない。追加パラメータに `--playground_v25` を書けば動く | 0% |
| フルファインチューン（`sdxl_train.py`）と、保存したチェックポイントへの `edm_mean` / `edm_std` | やっていない。フラグはエラーになる | 0% |

全体としては、LoRA の式は机上と CPU では揃っている。実物の重みと GPU が残っているので、完成度はおおよそ 40% と考える。

## 次にやること

1. 公式の safetensors で `tools/check_playground_v25_unet.py` を実行し、max abs diff を記録する。
2. `--playground_v25` で 10 step ほど学習し、VRAM と 1 step の秒数を見る。
3. 同じデータで `--pgv25_sigma_sampling=lognormal` と `--pgv25_loss_weighting=edm` を試し、既定のままにするか決める。Playground の報告（PDF 6 ページ）は、高解像度ではノイズを大きめに寄せたと書いている。数値は公開されていない。
4. ComfyUI で LoRA を載せる。ベースモデル側の `edm_mean` / `edm_std` で Playground と判定される。LoRA 自体にはそのキーは入らない。
5. 問題なければフルファインチューンに同じ式を渡す。

## ユーザーがやること

* このリポジトリを mogu-pc のワーカーに追加する。
* Windows の GPU マシンに公式の Playground v2.5 の重みを置く。
* 上の比較スクリプトと、短い学習をそのマシンで実行する。CPU の自動テストは重みが無いので比較を SKIP する。
* 16GB 向けの出発点は `HANDOFF.md` にある。学習率 `1e-5` の AdamW8bit も、Prodigy（学習率 1.0）も、まだ実測していない開始値である。
