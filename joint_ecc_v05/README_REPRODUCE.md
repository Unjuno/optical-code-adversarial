# 140 B 主ECC — 補助37セルのCRC検証済み情報で主LLRを補正する検証

## 実行条件
Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, OpenCV 4.13.0, x86_64。CPUはAMD EPYC 9V74、OpenBLAS/OMP 1 thread、batch 1、CPUクロック固定なし。GPU不使用。

## 先行結果への依存
`run_final.py --finalize` と `audit_final.py` は、既存方式の旧1,000件との1対1比較のため、次の絶対パスを読む。

```
/mnt/data/independent_140_v04/results/independent_0000_0499.csv
/mnt/data/independent_140_v04/results/independent_0500_0999.csv
```

パッケージには両CSVのコピー `baseline_archive/` を収録した。別環境で実行する場合、先に上記のパスへコピーする。もちろん元の旧実験の実行と、今回の新方式の実行は別である。

## シード
- 開発スクリーニング: payload `551100000+i` / camera `662100000+i` / i=0..399
- 開発確認: payload `771100000+i` / camera `882100000+i` / i=0..299
- 最終: payload `331100000+i` / camera `442100000+i` / i=0..999
- データ: `np.random.default_rng(payload_seed).bytes(142)[:140]`
- 行列: `d5_a2_s21`、行列ファイルSHA-256 `4d3f700cda39b60b4e1d2aee775f78ddd842a88a33718b98d55c7e5370a5e9bd`

## 実行
```bash
cd /mnt/data/joint_ecc_v05
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python development.py cache --stage screen
python development.py screen --stage screen
python development.py cache --stage confirm
python development.py screen --stage confirm
python run_final.py --chunk --start 0 --stop 500
python run_final.py --chunk --start 500 --stop 1000
python run_final.py --finalize
python audit_final.py
```

`FREEZE_SELECTION.json`に埋め込まれたソースのSHA-256を検査する。候補を変えたりファイルを変えた場合はそのまま実行できない（意図的な凍結）。

## 重要な解釈
このコードは、本体と補助の情報ビットを直接リンクさせない。復号に成功した補助37セルの既知RGB信号を新たな校正点として利用し、主LLRの信頼度を修正する「補助観測を用いた順次共同復号」であり、完全な因子グラフ同時MAP復号ではない。

性能は合成hard撮像チャネルに限定される。実カメラ評価、QR/JABの同条件比較は含まれない。
