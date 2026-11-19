# 140B専用・42bit追加パリティ行列の再選抜（実行報告）

**結果: 事前登録基準で PASS。ただし `hard` 99% は未達、カメラ実測とQR/JABの比較は未実施。**

## (a) 入力資料、環境、実際のコマンド

- GitHub リポジトリ `Unjuno/optical-code-adversarial`、コミット `4091ba8bd349518ea47d07078b91d048a6493277`。GitHub APIで `robust_code.py` と `final_validation.py` のblob SHA1を照合し、ZIP展開物の `git hash-object` と一致（`0be72d4a...` および `062cb8ae...`）。全ファイルのGitHubとZIPのバイト一致は確認していない。
- コンテナからの `git clone https://github.com/Unjuno/optical-code-adversarial.git repo` は `Could not resolve host: github.com` で失敗。代わりに `/mnt/data/optical_code_adversarial_v02.zip` を整合性検査し、 `/mnt/data/adv140_reselection/optical_code_adversarial_v02/` に展開。
- 研究ルート `/mnt/data/adv140_reselection/`、測定・行列ファイル `/mnt/data/adv140_reselection/outputs/`。
- Python 3.13.5、AMD EPYC 9V74、CPU 5個可視、クロック固定なし、NumPy 2.3.5、SciPy 1.17.0、OpenCV 4.13.0。 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`、 `cv2.setNumThreads(1)`、バッチサイズ1、GPUなし。撮像/符号化/復号は合成チャネル。OpenCVの幾何補正とNumPy/SciPyの浮動小数点動作を前提とする。Python ABIはCPython 3.13、x86_64、推論バックエンドは存在しない。

実行した主要なコマンド（前後のパス移動や `tee` ログ保存は `commands_executed.txt` に記載）:

```bash
cd /mnt/data/adv140_reselection
unzip -q /mnt/data/optical_code_adversarial_v02.zip
cd optical_code_adversarial_v02
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python final_validation.py --condition hard --name advgraph140 --n 1000
cd ..
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python aux42_lib.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python aux42_search.py --stage analytic
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python aux42_search.py --stage screen --start 0 --stop 7
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python aux42_search.py --stage screen --start 7 --stop 14
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python aux42_search.py --stage screen --start 14 --stop 16
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python aux42_search.py --stage select --start 0 --stop 2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python aux42_search.py --stage select --start 2 --stop 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python aux42_search.py --stage select --start 4 --stop 6
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python aux42_search.py --stage freeze
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python aux42_holdout.py --stage chunk --start 0 --stop 500
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python aux42_holdout.py --stage chunk --start 500 --stop 1000
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python aux42_holdout.py --stage finish
```

## (b) 新規ファイルと再選抜内容

- `PROTOCOL_PRE_REGISTERED.md`: 新行列の最終検証**前**に固定した実験条件と合否基準。
- `aux42_lib.py`: 旧版とビット一致する互換性試験、140B専用の追加42bitパリティ、符号化・軟判定24反復復号。
- `aux42_search.py`: 候補行列生成、CRC16有効データ全65,536通りの差分符号語によるHamming距離、開発セットによる2段選抜。
- `outputs/analytic_candidates.csv`, `outputs/screen_dev.csv`, `outputs/selection_dev.csv`: 16候補の解析・開発側実測成績。
- `outputs/selected_aux42_rows.npy`: 選択した行列の42×5インデックス表。SHA-256 `4d3f700cda39b60b4e1d2aee775f78ddd842a88a33718b98d55c7e5370a5e9bd`。
- `outputs/selected_aux42_H_42x74.csv`: 行列を42×74（各行は情報側5個＋自己パリティ1個の6個が1）として明示。
- `outputs/frozen_selection.json`: 最終検証前に確定した名前 `d5_a2_s21`、選択根拠・ハッシュ・凍結パラメータ。
- `aux42_holdout.py`, `outputs/holdout_pair_0000_0499.csv`, `outputs/holdout_pair_0500_0999.csv`: 1,000件のペア比較と試行別・原因別記録。
- `outputs/frozen_holdout_summary.json`, `outputs/failure_reason_transitions.csv`: 統計量、失敗理由の遷移、判定。
- `outputs/environment.json`, `outputs/*console.log`: バージョン・ログ。元のコードの結果ファイルは編集せず、展開コピー上でベースライン再実行。

**固定されたもの:** 270×270 RGB画像、37×37セル・4色、固定外枠、140byte (主138B+補助2B)、主ECC 18反復、補助ECC 24反復・α=0.8、検出閾値140、主パリティグラフ、配置、生成チャネル。**変更したものは、補助42本のパリティ検査行の情報側隣接先のみ。**

| 記号 | 意味 | SI単位 | 定義・定義域 | 型 |
|---|---|---|---|---|
| `u` | 補助データ+CRC16の32bit列 | 1 | `GF(2)^32` | 二値ベクトル |
| `P` | 42検査行×32情報ビットの係数 | 1 | `GF(2)^{42×32}` | 二値行列 |
| `I42` | 42次単位行列 | 1 | `GF(2)^{42×42}` | 二値行列 |
| `H` | 補助符号の全検査行列 | 1 | `H=[P | I42]` | 42×74二値行列 |
| `c` | 74bit補助符号語 | 1 | `c=[u; Pu]`（二元体演算） | 二値ベクトル |

`Hc=Pu+Pu=0` (GF(2))。`I42` ブロックがあるため `rank(H)=42`。ただし「どんなカメラ条件でも必ず復号」という保証ではない。1セル2bit×37セル=74bit、32情報bitの内16bitはCRC16、実ペイロードは16bit=2byte。全項は無次元で単位整合。

開発用スクリーンは候補16個を各 hard 200 + challenge 90件、上位5候補+旧行列は別の開発データで hard 700 + challenge 300件評価した。開発の hard `payload_seed=341000000+i`、`channel_seed=441000000+i`、challenge `541000000+i`。**選択には以下の最終シードを使っていない。**

選抜対象 `d5_a2_s21` は、開発後段でhard の補助復号 **690/700**（旧 **679/700**）、challenge **289/300**（旧 **276/300**）。CRC有効な全2byteペイロードに対する最小Hamming距離は旧 **14**、新 **19**。Tannerグラフ4-cycle数は旧190、新24。これは構造的な比較であり、ソフト復号成功率を直接証明するものではない。

## (c) 凍結した同じ1,000件での完全復元・補助CRC・Wilson 95%信頼区間

評価対象: `hard` 1回撮影、`payload_seed=121100000+i`、`camera_seed=222100000+i`、`i=0..999`。旧方式の成功ビット列は保存されている `outcomes_hard_advgraph140.npy` と全1,000件完全一致した。

| 指標 | 旧 `advgraph140` | 新 `aux42 d5_a2_s21` |
|---|---:|---:|
| 140byte完全一致 | **928/1000 (92.8%)** | **939/1000 (93.9%)** |
| 完全一致率 95% Wilson | 91.03–94.24% | 92.24–95.22% |
| 補助CRC失敗 | **13/1000** | **6/1000** |
| 補助CRC率 95% Wilson | 0.761–2.211% | 0.275–1.303% |
| 主パリティ失敗 | 40 | 37 |
| 主CRC失敗 | 13 | 12 |
| 検出・色校正失敗 | 6 | 6 |
| 平均完全回収容量 | 129.92byte/試行 | 131.46byte/試行 |
| 復号レイテンシ中央値 / P95 | 5.68 / 8.98ms | 5.38 / 8.70ms |

成功件数の対応表は、双方成功924件、双方失敗57件、旧のみ成功4件、新のみ成功15件。対応ありの成功率増分は **+1.1ポイント**、対応ありbootstrap（10,000再標本）の95%区間は **+0.3〜+1.9ポイント**。全成功に対する片側exact McNemar（不一致ペアでの二項検定）は `p=0.0096054`。

**事前登録された主要検定は「同一の主復号合格条件にある試行の補助CRC失敗」**。共通して主復号に合格した939件の中で、旧のみ補助CRC失敗9、新のみ失敗2、双方失敗4。片側の対応あり正確二項検定は **p=0.03271484375**。この検定は事前登録の有意水準0.05未満。

**原因の解釈:** 追加CRC失敗の減少は7件、総成功増は11件。新しい補助セル模様が撮像・幾何補正にも影響し、主パリティ/主CRC失敗もわずかに減った。**11件すべてを補助ECCの純粋な訂正力改善とみなすのは誤り**。

## (d) 事前登録基準による判定

`PROTOCOL_PRE_REGISTERED.md` のルール: 新方式の全体成功件数が旧方式より大、追加CRC失敗が旧より少、共通主合格例での片側対応検定 `p<0.05`、その他の実装条件が凍結されていれば **PASS**。失敗数等が悪化ならFAIL、改善しても有意でなければUNCERTAIN。

今回: `939>928`、`6<13`、`0.0327148<0.05`、凍結した仕様を維持したので、**この限定シミュレーション・事前登録基準では PASS**。

一方、元の実用目標「hard で完全一致99%以上」に対しては **FAIL (93.9%)**。1,000件での95% Wilson上限でも99%には届かない。QR/JABや実カメラを上回るという主張は未検証。

## (e) 実行できなかった手順と理由

1. GitHubからの `git clone`: DNSによるホスト解決失敗。**ZIPを代替使用**。GitHubのAPIからcommitと `robust_code.py`、`final_validation.py` のblob SHAを取得し、ZIP内該当ソースとの一致を別途確認。
2. 実カメラ/印刷、規格準拠QR・JABとの比較: 物理デバイス・画像・比較実装のないコンテナでは実行していない。今回の局所的な追加パリティ研究の合否に含めない。
3. 元の `train.py` / `optimize_core_graph.py` の再訓練: **意図的に未実施**。140B追加パリティ以外を変更しないという実験上の固定条件を満たすため。
4. `docs/DECLARATION.md` はGitHubの当該commitから読めるが、添付ZIPには含まれていない。ZIP内のコードと結果を使用した。GitHubリポジトリへのpushは依頼されておらず実施していない。
5. 開発用スクリーンの最初の一括起動は時間制限で10候補までで中断された。結果を最終選抜に利用せず、**同条件を7+7+2候補に分割して再実行し、16候補すべて終了した**。最終検証1,000件は500+500件で完全実行した。

## ERROR CHECK と科学的留意点

- 旧方式の正答ビット列1,000件が既存の保存結果と完全一致。旧の失敗内訳40/13/13/6も一致。
- 本体の補助37セル以外の符号画像が変わっていないことを20ランダムメッセージで検証。300個のCRC/パリティ式および無雑音ラウンドトリップを検証。CRC16有効な全65,536通りの差分符号語に関する最小距離を厳密列挙し、200ペアで再確認。
- 今回使ったholdoutの1,000件は以前から公開されていた旧性能に対応するシードであり、**完全に未知の新規ベンチマークではない**。新しい行列の選択にはこの1,000件の個別結果を使用していないが、p値はこれらの条件と事前に設定した探索手順に限定して解釈する。
- CPUクロックは固定していないため数百µs程度の速度差を高速化の証拠とみなさない。最大の改善を主張できるのは**同じ合成撮像分布での完全復元率**と**追加CRC失敗率**。
- 学際的意義: 符号理論（疎なパリティ検査行列）、組合せ最適化（CRC有効符号語の距離・4-cycle）、コンピュータビジョン（撮像と色校正の結合）、統計（対応ありの推定と検定）。

## 次に考える問い

**同じ補助パリティ行列を学習していない撮影機種・未知の光学分布に適用しても、この7件の追加CRC失敗減少は再現されるか？** 次段階は新しい独立シードと実カメラ条件での検証である。
