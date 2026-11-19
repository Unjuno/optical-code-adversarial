# 実際に実行した主コマンド（この順番）

## 初期環境・ソース
- 入力: `/mnt/data/independent_140_v04/adv140_reselection/`
- 作業ディレクトリ: `/mnt/data/joint_ecc_v05/`
- `cp` で `aux42_lib.py`、`robust_code.py`、`lab_v01.py`、`selected_aux42_rows.npy`、`frozen_selection.json` だけを作業ディレクトリにコピー。最終1,000件の旧個票は候補選抜のためにコピー・閲覧していない。
- Pythonを使用して `PROTOCOL_PRE_REGISTERED.md`、`joint_receiver.py`、`development.py`、`run_final.py` を新規作成。
- `python joint_receiver.py` による通過検査。
- 開発側70件で旧復号器との理由・元バイト列の完全一致を検査（70/70）。

## 候補選抜（最終1000件より前）

```bash
cd /mnt/data/joint_ecc_v05
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python development.py cache --stage screen 2>&1 | tee logs/dev_cache_screen.log
python development.py screen --stage screen 2>&1 | tee logs/dev_screen.log
python development.py cache --stage confirm 2>&1 | tee logs/dev_cache_confirm.log
python development.py screen --stage confirm 2>&1 | tee logs/dev_confirm.log
```

候補32件から6件を別の開発シード300件で確認し、候補名 `affine_ridge4_covidentity_beta0.5` を採用。SHA256を含む `FREEZE_SELECTION.json` は最終シードで画像を作る前にPythonで生成して固定。

## 最終評価（凍結後）

```bash
cd /mnt/data/joint_ecc_v05
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python run_final.py --chunk --start 0 --stop 500 2>&1 | tee logs/final_0000_0499.log
python run_final.py --chunk --start 500 --stop 1000 2>&1 | tee logs/final_0500_0999.log
python run_final.py --finalize 2>&1 | tee logs/finalize.log
python audit_final.py 2>&1 | tee logs/audit_final.log
```

## 事後監査に使った旧個票
`/mnt/data/independent_140_v04/results/independent_0000_0499.csv`
`/mnt/data/independent_140_v04/results/independent_0500_0999.csv`

候補選抜時・設定凍結前に、これらの旧個票は読み込んでいない。集計後、比較のために `baseline_archive/` にコピーした。
