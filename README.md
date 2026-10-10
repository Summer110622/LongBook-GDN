# LongBook-GDN

[LongBook-GDN](https://github.com/Summer110622/LongBook-GDN) は、Longbook27 向けの
ローカル GGUF クライアントを開発するリポジトリ。
**クライアント名は Longbook27、実行ファイル名は `longbook27`。C++17 製の CLI と、必要な
モデル実装だけを残した独自の llama.cpp エンジンを同梱する。**

対象は [summerMC/Qwen3.8-27B-Longbook27-VL-GDN64-GGUF][model]。
クライアントは `libllama` に直接リンクし、モデルの読み込み、入力、チャット、
token budget、生成、中断を扱う。実装の状態は **v0.1.0 開発版**。
CPU の合成モデル推論と実 GGUF のヘッダー認識を確認済み。
**実際の 27B チェックポイントの生成、CUDA / Metal、VL は未検証。**

## 開発の開始点

実装に先立ち [AGENTS.md](AGENTS.md)、[PR ポリシー](docs/PR_POLICY.md)、
[PR テンプレート](.github/pull_request_template.md) を作成した。
既存の `main` の履歴を保ち、目的ごとのブランチから PR で変更を取り込む。
GitHub Actions の実行結果は各 PR の **Checks** を確認する。
[検証記録](docs/VERIFICATION.md) はローカルでの実行結果を記したもので、リモート CI の結果とは区別する。
workflow とポリシーの追加では ruleset は自動設定されない。管理設定はリポジトリ管理者が別途行う。

## 実装した機能

- ローカル GGUF のヘッダー・層構成を検査してからモデルをロード。
- 単発生成、行単位の対話、UTF-8 ファイル / 標準入力。
- 対象モデルの text-only chat template、system message、thinking on/off。
- 分割 prefill、CPU thread 数、GPU layer 数、sampling / seed の指定。
- 文脈超過をエラーにし、入力の自動切り捨てや recurrent state のシフトをしない。
- `/reset`、`/exit`、EOF、SIGINT / SIGTERM。GPU 中断は batch の境界で反映される場合がある。
- stdout に生成本文、stderr に操作案内・診断・token 数・時間を出力。
- 重みの SHA-256 を明示的に検証する、Python 標準ライブラリの検査ツール。

## なぜ専用のエンジンが必要か

公開 [モデルカード][card] は `vnlpscale/llama.cpp` を対応先としている。
実 GGUF では metadata と tensor の構成に次の相違がある。

| 項目 | metadata | 実 tensor |
| --- | --- | --- |
| 層数 | `block_count=65`、`nextn_predict_layers=1` | 主層 64、MTP tensor なし |
| Attention | `full_attention_interval=4` | 64 層すべて recurrent / GDN |
| Modality | repository 名に `VL` | vision tensor と mmproj はない |

エンジンは [c5709247c243a0b1ae299ddb9d8f4c8b0279be89][engine] に固定した。
tensor の存在から topology を判定し、欠落 MTP と全 recurrent graph を扱う
[モデル対応修正][fix] を保持している。通常の upstream `master` に差し替えない。
出典と詳細は [MODEL.md](docs/MODEL.md) に記録した。

## ビルド

必要なものは CMake 3.21 以上、Ninja、C/C++17 compiler、Python 3.10 以上。
エンジンのソースは同梱済みで、通常の configure / build / inference はネットワークを使わない。
モデルのダウンロードは別の明示的な操作として行う。

### リポジトリの取得

```bash
git clone https://github.com/Summer110622/LongBook-GDN.git
cd LongBook-GDN
```

以下のコマンドはリポジトリのルートで実行する。レビュー中の変更を試す場合は、
対象 PR に表示されるブランチを checkout してからビルドする。

### Linux / CPU

```bash
cmake --preset cpu
cmake --build --preset cpu --parallel 4
ctest --preset cpu

./build/cpu/longbook27 --version
```

この x86 CPU preset は SSE4.2、AVX、AVX2、FMA、F16C、BMI2 を前提にする。
確認した環境は Linux x86_64 / GCC 13.3 / CMake 4.4.4 / Python 3.12。
GPU はこの検証環境に存在しない。

### NVIDIA / CUDA

ローカルに CUDA toolkit と対応 compiler / driver を用意する。

```bash
cmake --preset cuda
cmake --build --preset cuda --parallel 4

./build/cuda/longbook27 -m /path/to/SpeedX27-GDN64-Q4_K_M.gguf \
  --interactive --gpu-layers 99
```

CUDA source と preset は同梱しているが、この環境ではビルド・推論を確認していない。
VRAM に収まらない場合は `--gpu-layers`、batch / ubatch を調整する。
重みのファイルサイズだけで実行時 RAM / VRAM の必要量は決まらない。

### Apple Silicon / Metal

```bash
cmake --preset metal
cmake --build --preset metal --parallel 4
```

Apple の開発環境が必要。Metal も未検証。
Windows の対応をリリースとして保証する前に、別途 compiler / CUDA / console の検証を行う。

## モデルを用意する

公開されているファイル名は `SpeedX27-GDN64-Q4_K_M.gguf`。
名前を Longbook27 に推測して置き換えない。
[Hugging Face CLI][hf-cli] を利用する場合の取得例は以下。

```bash
hf download \
  summerMC/Qwen3.8-27B-Longbook27-VL-GDN64-GGUF \
  SpeedX27-GDN64-Q4_K_M.gguf \
  --revision f3ee816de60fd7f67911e949a53514b4b17f84bb \
  --local-dir ./models-cache

python3 tools/model_info.py \
  ./models-cache/SpeedX27-GDN64-Q4_K_M.gguf --verify-sha256
```

重みは **16,746,122,528 bytes（約 15.596 GiB）**。
公開 SHA-256 は
`5af94bfd3e1725ab09d8b82ef929706362bc0b269a7858d893389f9a40b27521`。
revision、checksum、tokenizer、各層の形状と量子化を [model manifest](models/longbook27.json)
に固定した。重みはソースにも配布アーカイブにも含めない。

ヘッダーだけを検査する場合は以下を使う。

```bash
./build/cpu/longbook27 \
  -m ./models-cache/SpeedX27-GDN64-Q4_K_M.gguf --inspect

python3 tools/model_info.py \
  ./models-cache/SpeedX27-GDN64-Q4_K_M.gguf --json
```

native の `longbook27_signature` は層数と主要 metadata の検査であり、
Python inspector の厳密な全 descriptor / tokenizer 検査とは範囲が違う。
`tensor_extents_fit_file` は tensor の範囲がファイルサイズ内に収まることだけを示す。
**どちらも重みの同一性を証明しない。** 同一性の確認には `--verify-sha256` を使う。

## 利用例

### 単発の質問

```bash
./build/cpu/longbook27 \
  -m ./models-cache/SpeedX27-GDN64-Q4_K_M.gguf \
  -p "Gated DeltaNet の recurrent state を説明してください。" \
  -n 512 --thinking off
```

### チャット

```bash
./build/cpu/longbook27 \
  -m ./models-cache/SpeedX27-GDN64-Q4_K_M.gguf \
  -i --system "日本語で、前提と結論を区別して答えてください。"
```

1 行が 1 回の入力となる。`/reset` は履歴を消し、`/exit` と EOF は終了する。
複数行の文書は `--prompt-file` で入力する。

### 長文ファイル

```bash
./build/cpu/longbook27 \
  -m ./models-cache/SpeedX27-GDN64-Q4_K_M.gguf \
  --prompt-file ./book.txt \
  --system "文書を読み、章ごとの論点と根拠を整理してください。" \
  --ctx-size 32768 --predict 1024 --batch-size 512 --ubatch-size 128
```

ファイルは最大 64 MiB、UTF-8 テキストを想定する。
PDF / EPUB の直接読み込みや OCR は実装していない。
`--prompt-file -` は標準入力を EOF まで読み込む。

## 会話・長文の仕様

入力を chat template で整形した後の token 数を `T`、生成上限を `N`、
指定 context を `C` とすると、クライアントは次の条件を要求する。

```text
T + N <= C <= min(GGUF context_length, 262144)
```

`T` には system message、過去の会話、chat template の token が含まれる。
満たさない場合は推論を開始せず、理由を stderr に返す。
metadata の 262,144 tokens は、このクライアントで全長を実測した保証ではない。

GDN の状態を任意の位置で編集すると不整合を招くため、現段階では **毎ターン
新しい context を作り、履歴全体を prefill** する。正しさを確認しやすい一方、
会話が長いほど再処理時間が増える。効率的な persistent state / rollback は今後の独立 PR とする。

`--thinking off` が初期値。`on` は `reasoning_effort=medium` 相当とし、
原本の `xhigh` 用 system instruction を追加しない。
過去の assistant は reasoning と answer を分けて保存する。
本文両端の ASCII 空白を template に合わせて除く。
非 ASCII の両端空白は保持するため、Jinja 全機能・Unicode trim の完全互換は主張しない。
tools / images / video の template は扱わない。
`--raw` は単発の直接 tokenization で、system message と thinking の整形を行わない。

## エンジンの削減

| 集計対象 | 元 source archive | この配布 source |
| --- | ---: | ---: |
| 通常ファイル数 | 3,450 | 515 |
| 非圧縮 bytes | 166,175,683 | 11,033,319 |

元 source の約 **93.35%** を除外した。生成した audit JSON は上記のファイル数に含めない。
これは source tree の削減であり、実行速度や RAM / VRAM の削減率ではない。

`src/models/` では `qwen35.cpp`、`delta-net-base.cpp`、共有宣言のみを残す。
factory も他 architecture を拒否する。server / Web UI / converters / training tools /
common / multimodal / 不要 backend を除外し、GGUF / tokenizer / sampler /
GDN / recurrent / hybrid memory と CPU / CUDA / Metal の source を保持した。
共有コアには未使用の宣言や演算も残る。さらに細かい kernel 削減は依存関係を検証する別段階とする。

```bash
# 同梱 source の整合性を検証。ネットワーク不要。
python3 scripts/vendor_engine.py --verify

# 明示的なエンジン再生成。モデル重みは取得しない。
python3 scripts/vendor_engine.py --fetch --replace
```

固定 archive SHA-256、変更前後の file hash、patch、削除ファイル一覧を保存し、
対象が変わった場合は自動で推測して patch を当てずに失敗させる。
詳細は [ENGINE.md](docs/ENGINE.md)。

## 検証と残る作業

[検証記録](docs/VERIFICATION.md) に結果と再現コマンドをまとめる。
fast tests では約 300 KiB の合成 F32 GGUF を生成し、実際の CPU graph を実行する。
全 recurrent / obsolete interval / absent MTP、分割 prefill、文脈境界、
UTF-8、chat reset、終了処理を確認する。合成データの結果を品質評価として使わない。

次のリリース条件は、実 27B Q4_K_M でのロード・生成・複数ターン・長文境界・中断、
対象 GPU backend の実測。VL には互換 mmproj の特定と独立した実装・検証が必要。

## ライセンス

Longbook27 本体は [MIT](LICENSE)。エンジンの元ライセンスと著作権表示は
`third_party/llama.cpp/` に保持している。モデルカードは重みを Apache-2.0 と表示している。
コードと重みの license / provenance は別に扱う。

[model]: https://huggingface.co/summerMC/Qwen3.8-27B-Longbook27-VL-GDN64-GGUF
[card]: https://huggingface.co/summerMC/Qwen3.8-27B-Longbook27-VL-GDN64-GGUF/raw/f3ee816de60fd7f67911e949a53514b4b17f84bb/README.md
[engine]: https://github.com/vnlpscale/llama.cpp/tree/c5709247c243a0b1ae299ddb9d8f4c8b0279be89
[fix]: https://github.com/vnlpscale/llama.cpp/commit/09b4a507062509b021ed07860fdf5dbc6727899f
[hf-cli]: https://huggingface.co/docs/huggingface_hub/guides/cli
