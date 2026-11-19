# 次世代二次元コード — 敵対的探索 v0.2

**研究対象は二次元コード本体**。モデルの重みに情報を保存したり、外部URLを参照してデータを復元したりしない。RGB4の37×37セル記録領域、外側4セルの余白を含めた最終画像は270×270 RGB画素。**実験的な独自符号であり、QR/JAB規格に準拠しない。**

## 今回の改善

- 広い光学歪み範囲での失敗条件を、ブラックボックスの反例生成器が探索。ガウスぼけ、ピクセル・セルノイズ、射影ずれ、回転、色混合、白平衡を変化させる。
- データ追加のための**外枠着色を廃止**。外枠は黒いまま、前バージョンで使っていなかった37個の内側セルを追加容量に利用。
- 37セル×RGB4=74 coded bits。堅牢モードは **2 B＋CRC16＋42パリティbit** → 合計140 B。高容量モードは**4 B＋CRC16＋26パリティbit** → 合計142 B。
- 1,152個の主データセルは従来通りの2 bit/cell、主符号1/2。**同じ符号化率のまま**LDPC風の疎パリティ検査グラフを敵対的学習用の反例で探索。
- 局所位置検出は輪郭とRGB基準パイロットで行い、カメラ姿勢は与えない。SHA-256は試作例および複数符号ファイル復元の最終検証に利用。

## 結果（2026-11-15）

| 方式 | 容量 | `medium` 完全一致 /1,000 | `hard` 完全一致 /1,000 | `hard` 検証済みB/試行 | `hard` P50 / P95 復号ms |
|---|---:|---:|---:|---:|---:|
| v0.1固定外枠 | 138B | 1000 | 642 | 88.60 | 4.80 / 7.95 |
| v0.1着色外枠 | 140B | 994 | 82 | 11.48 | 4.45 / 8.11 |
| v0.2予備37セル (旧LDPC) | 140B | 未測定 | 705 | 98.70 | 4.98 / 8.00 |
| **v0.2敵対的選択グラフ** | **140B** | **1000** | **928** | **129.92** | **5.01 / 7.97** |
| v0.2敵対的グラフ＋配置 | 142B | **1000** | 846 | 120.13 | 5.01 / 8.17 |

公平性: すべて270×270 RGB画像、同一`medium`/`hard`カメラ条件、同じペイロードと撮像乱数seed、コード全体の完全一致で成功。`hard`は設計仮定で、実世界の性能を意味しない。表の `未測定` は独立した最終1,000試験を行っていないことを示す。

さらに、完全未知の120個のメッセージに対して8個ずつの **敵対的近傍視点（合計960画像/方式）** を生成し、8画像すべてで完全一致できるメッセージ数は、v0.1固定外枠43/120、v0.2敵対的140B **102/120**、v0.2高容量142B 82/120となった。これは有限の8条件テストであって連続的な最悪条件保証ではない。

**実務判定: 140Bは堅牢モード、142Bは中程度以下向け容量モード。`hard`成功率92.8%は99%の信頼性目標未達。SOTA未実証。**

## 実行

Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, OpenCV 4.13.0, matplotlib/Pillow。`requirements.txt`参照。

```bash
cd optical_code_adversarial_v02
python -m pip install -r requirements.txt
python release_validation.py
python offline_file_demo.py
# 1,000 independent heldout simulations per mode
OPENBLAS_NUM_THREADS=1 python final_validation.py --condition hard --name base138 --n 1000
OPENBLAS_NUM_THREADS=1 python final_validation.py --condition hard --name advgraph140 --n 1000
OPENBLAS_NUM_THREADS=1 python final_validation.py --condition hard --name combined142 --n 1000
# worst-of-8 stress with identical channel seeds across methods
OPENBLAS_NUM_THREADS=1 MODE=advgraph140 python stress_test.py
# adversarial training (randomness pre-fixed), optional
OPENBLAS_NUM_THREADS=1 python train.py
OPENBLAS_NUM_THREADS=1 python optimize_core_graph.py
python make_figures.py
```

**重要:** 再実行時にはGPUを使用しない。AMD EPYC 9V74（仮想環境で5 CPU visible）、CPU、バッチサイズ1、クロック非固定。`OPENBLAS_NUM_THREADS=1`推奨。画像1枚の符号化・撮影時間は復号時間に含まない。Python ABI／ライブラリバージョン・OpenCV画像処理で結果が変わり得る。

## ファイル一覧

- `lab_v01.py`：再現する旧ベースラインの符号化器・撮像器・復号器。
- `robust_code.py`：140B/141B/142Bコードと主ECCの選択、追加37セルのECC。
- `adversary.py`：撮像失敗を探す擾乱探索器と符号配置候補。
- `train.py`：40例の敵対的生成セットで13レイアウトを選抜。
- `optimize_core_graph.py`：21個の主ECCパリティ検査グラフを独立開発シードと敵対的生成セットで評価。
- `final_validation.py`：最終未使用シードで完全一致評価。
- `stress_test.py`：各メッセージ8つの敵対的撮像条件でのストレス試験。
- `dense_aux.py`：RGB8の追加容量拡大。性能悪化のため不採用。
- `release_validation.py`：データ復元、配置の重複、PNGラウンドトリップ、SHA256確認。
- `offline_file_demo.py`：2,048 Bの任意ファイルを17個のコード画像に分割・順不同復元・SHA256で検証。
- `results/benchmark_summary.csv`：成績とWilson信頼区間。
- `results/layout_training.csv`, `results/core_graph_training.csv`：設計候補選択の詳細。
- `results/holdout_final_*.csv`, `results/outcomes_*.npy`：各1,000件の独立検証と二値成功列。
- `results/stress_*.csv`、`results/training_attacks.json`、`results/holdout_attacks.json`：生成された敵対的条件、結果。
- `results/actual_symbols_comparison.png`：実際に生成されたコードの比較。`results/hard_success.png`・`hard_goodput.png`・`latency_p95.png`：定量的グラフ。
- `results/example_robust140.png`, `example_robust140_captured.png`, `example_capacity142.png`：PNG検証例。
- `RESEARCH_REPORT_JA.md`：理論、統計検定、限界、H/T/D/C/U。

## 注意

- 完全一致率1000/1000は成功確率100%の証明ではない（両側95% Wilson下限約99.62%）。
- 敵対的生成は**条件パラメータのブラックボックス最適化**であり、GANの学習ではない。連続空間の厳密なminimax最適解ではない。
- 光学チャネルは合成モデル。実印刷・実カメラ、ロールシャッター、色域差、ISP、逆光などは未評価。
- 比較相手は同じ試作の旧仕様であって、ISO QR / JAB Codeとの公平な直接比較ではない。
- 容量モードは送受双方で指定する。符号内部にモード識別子を記録していない。
- 付随するCRC16/32は暗号学的な認証に不適切。ファイル単位の完全一致はSHA-256で確認。

## 原論文

- Gallager, R. (1962), *Low-Density Parity-Check Codes*, DOI: 10.1109/TIT.1962.1057683。
- Athalye et al. (2018), *Synthesizing Robust Adversarial Examples*, ICML, https://proceedings.mlr.press/v80/athalye18b.html（変換条件の反例最適化という考え方）。
- Kurmukova and Gündüz (2024), *Friendly Attacks to Improve Channel Coding Reliability*, AAAI, https://ojs.aaai.org/index.php/AAAI/article/download/29230/30321（符号信号の頑健性改善）。
- JAB Code: ISO/IEC 23634:2022, https://www.iso.org/standard/76478.html 。
