# Longbook27 専用エンジン

Longbook27 は固定した `vnlpscale/llama.cpp` から必要なソースを抽出し、
独自の C++17 実行ファイルから `libllama` を直接呼び出す。
モデル読み込み、トークン化、prefill、生成、停止、会話履歴の管理は
`src/main.cpp` と `src/client.cpp` が担当する。

## 固定した出自

| 項目 | 値 |
| --- | --- |
| 対応 fork | <https://github.com/vnlpscale/llama.cpp> |
| commit | `c5709247c243a0b1ae299ddb9d8f4c8b0279be89` |
| 元プロジェクト | <https://github.com/ggml-org/llama.cpp> |
| 全層 GDN 対応の主要修正 | `09b4a507062509b021ed07860fdf5dbc6727899f` |
| source archive | `https://codeload.github.com/vnlpscale/llama.cpp/tar.gz/c5709247c243a0b1ae299ddb9d8f4c8b0279be89` |
| archive bytes | `36,924,919` |
| archive SHA-256 | `29af4bbc9052f1bf5f730ca7cc27c1bf54827786ab712b3d1bdeed820482927c` |
| 保持ファイル manifest SHA-256 | `5cb22708452455a326da803cced86be1d32f1da47d63eea5452ba947f16f7898` |

`engine.lock.json` が機械可読な固定情報を持つ。snapshot は
`third_party/llama.cpp/` に同梱され、初回の configure/build でも取得処理はない。
モデルの重みはこの source archive と別であり、一切含めない。

### 対応 fork を使う理由

[固定 fork の Qwen35 実装](https://github.com/vnlpscale/llama.cpp/blob/c5709247c243a0b1ae299ddb9d8f4c8b0279be89/src/models/qwen35.cpp)
は、明示的な recurrent-layer 配列がない場合に各層の
`blk.N.ssm_conv1d.weight` から topology を判定する。
全層 recurrent の場合は不要な attention-cache / position graph input を作らない。
また、チェックポイントに存在しない optional MTP tensor の処理を維持する。
指定 GGUF の詳細は `models/longbook27.json` を参照する。

全層 GDN でも `llama_memory_hybrid_context` を内部で利用するため、
hybrid memory / recurrent state / 必要な KV abstraction は残している。
型名やディレクトリ名だけを根拠にこれらを削除しない。

## 実際のソース削減

抽出元 archive の regular file を数えた結果は次のとおり。
「除外」は独自配布に含めないことを意味し、元リポジトリは変更しない。

| 区分 | ファイル数 | bytes |
| --- | ---: | ---: |
| 元 archive の全 regular file | 3,450 | 166,175,683 |
| 保持したファイル・patch 前 | 515 | 11,045,690 |
| 保持したファイル・patch 後 | 515 | 11,033,319 |
| 配布から除外 | 2,935 | 155,129,993 |
| 保持した C/C++/CUDA/Objective-C/Metal ソース・header | 484 | 10,801,156 |
| 除外した同ソース・header | 903 | 22,756,314 |

元ソースツリーの bytes を **93.35%** 除外している。
この割合には upstream の UI asset、文書、変換用コード等も含む。
実行時メモリ、実行ファイルのサイズ、推論速度の改善率ではない。
生成した監査 JSON 自身は上の集計に含めない。

`third_party/llama.cpp/LONGBOOK27_VENDOR_AUDIT.json` に全保持ファイルの
元 SHA-256 / patch 後 SHA-256 / bytes、および全除外パスと bytes を保存する。
これによりビルド対象を無効化したことと、ソースそのものを除外したことを区別できる。

### 保持する範囲

- GGUF loader、mmap、tokenizer、sampler、必要な graph / memory 共通実装。
- モデル実装は `qwen35.cpp` と共有 `delta-net-base.cpp` のみ。
- `models.h` の共通宣言。未使用モデルの宣言は残るが、実装と factory は除外する。
- GGML 共通層、CPU、CUDA、Metal の source。default build は CPU のみ。
- ルートの `LICENSE`、`AUTHORS`、`licenses/`、保持 source 内の copyright/license 表記。

### 除外する範囲

- `common/`、HTTP server、Web UI、downloader、upstream の executable。
- `tools/`、`examples/`、`pocs/`、upstream の test runner。
- モデル変換、量子化 CLI、学習関連の utility。
- 別アーキテクチャのモデル実装、未選択 GPU backend、RPC backend。
- 画像・音声・動画 encoder と関連 vendor helper。

量子化済み tensor の読み取り・演算コードは必要なため残す。
クライアント独自のテストはプロジェクト直下の `tests/` で管理する。

## 独自 patch

`patches/0001-curate-qwen35.patch` はレビュー用の通常の unified diff。
同じ変更を `patches/0001-curate-qwen35.json` の exact replacement で再現する。
両方の digest を lock に固定し、生成時に両者が一致することを検査する。

| 対象 | 変更 |
| --- | --- |
| `src/CMakeLists.txt` | glob をやめ、`qwen35.cpp` と `delta-net-base.cpp` を列挙する。 |
| `src/llama-model.cpp` | model factory が `LLM_ARCH_QWEN35` 以外を明示的に拒否する。 |
| ルート `CMakeLists.txt` | 不要な vendor helper の無条件 build を削除する。 |
| `ggml/CMakeLists.txt` | 外側の application Git SHA ではなく、固定 engine SHA を build 情報に使う。 |

patch は元ファイル全体の digest、変更前の文字列の一意一致、変更後の digest を検査する。
別 revision に曖昧に適用することはない。

## ビルドと検証

必要なツールは CMake 3.21 以上、Ninja、C/C++17 compiler、Python 3.10 以上。
CPU default は外部パッケージを自動取得しない。CUDA にはローカル CUDA toolkit、
Metal には Apple 環境と Xcode のツールが必要。

```sh
python3 scripts/vendor_engine.py --verify
cmake --preset cpu
cmake --build --preset cpu --parallel 2
ctest --preset cpu
```

CMake の configure 時にも snapshot digest を検証する。
CTest は CLI help/version、Python unit/native tests、出自の検証を実行する。
`LONGBOOK27_BIN` を CTest から native tests に渡すため、合成 fixture が
実際のクライアント executable を通る。

CUDA / Metal は `cpu` の代わりに `cuda` / `metal` preset を使う。
これらはソースとビルド入口を用意した段階であり、backend 上の実推論結果は別途記録する。
現在の x86 CPU preset は upstream default により SSE4.2、AVX、AVX2、
FMA、F16C、BMI2 を有効にする。生成された x86 binary はこれらの命令を
備えた CPU を前提とする。`LONGBOOK27_NATIVE=OFF` は build machine 専用の
最適化を抑止する設定であり、これらの ISA default を無効化する設定ではない。

公開 API は固定 fork の `llama_batch` / `llama_decode` を使う。
新しい upstream の `llama_batch_ext` / `llama_process` API と混ぜない。
`llama_sampler_sample` は accept まで含むため、同じ token を二重 accept しない。

## snapshot の再生成

ローカル archive を使用する場合：

```sh
python3 scripts/vendor_engine.py --archive /path/to/llama-c5709247.tar.gz --replace
```

明示的にソースを取得する場合：

```sh
python3 scripts/vendor_engine.py --fetch --replace
```

既存 snapshot がなければ `--replace` は不要。
既存ファイルにローカル編集がある場合は、上書きせずエラーにする。
archive のサイズと SHA-256 を検査してから展開し、パスの逸脱、重複、link、
特殊ファイル、過大サイズを拒否する。`extractall` は使わず、allowlist に含む
regular file だけを新しい作業ディレクトリへ書き出す。

engine 更新時は個別の PR で source pin、archive digest、allowlist、patch、
保持ファイル manifest digest を更新する。旧 digest と一致しない場合は処理を止め、
実測した新 digest をエラーに表示する。内容をレビューしてから lock を更新し、
同じ archive から再生成できることを確認する。

## 検証範囲と残る作業

高速なテストは **4 recurrent blocks と宣言だけの MTP 1 block** を持つ小さな
合成 F32 GGUF で tokenizer、GDN graph、state 更新、prefill 分割等を検証する。
実モデルの **64 main recurrent layers** は、実 GGUF の header / tensor descriptor で
構成を確認した。実際の 27B チェックポイントの weight loading / 生成は未実施であり、
合成モデルの結果は実モデルの品質・速度・長文保持能力を示すものではない。
実施した検証と再現条件は [VERIFICATION.md](VERIFICATION.md) に記録する。
実モデル、CUDA、Metal、対応 projector を使った VL の受け入れ試験は
`docs/PR_POLICY.md` の条件に従って別に記録する。

## ライセンス

保持した fork のルート `LICENSE` は MIT、copyright は
`2023-2026 The ggml authors`。`AUTHORS` と元 source 内の通知を保存する。
同梱 `ggml/` に独立した `LICENSE` ファイルはない。
`cmake --install` は Longbook27 本体の `LICENSE` と、engine の
`LICENSE` / `AUTHORS` / `licenses/` / 出自情報を `share/longbook27/` へ配置する。
Mozilla の SGEMM、および YaRN の source header 内にある追加通知も
`ENGINE-THIRD-PARTY-NOTICES.txt` として生成し、installed binary に添付する。
保持 source の一部に埋め込まれた個別の copyright/license header も原文のまま
source 配布に含める。モデル重みのライセンスとエンジンのライセンスは別々に扱う。
