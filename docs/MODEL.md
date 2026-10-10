# 対象モデルと互換性

このクライアントが対象とするファイルは **`SpeedX27-GDN64-Q4_K_M.gguf`**。
リポジトリ名は Longbook27 だが、公開ファイル名には SpeedX27 が残っている。
名前から構造を推測せず、固定した revision のメタデータと tensor table を使う。
以下は 2026-10-10 に取得した一次情報と、GGUF の header / descriptor の直接検査による。
チェックポイントをロードした推論結果ではない。

## 固定したファイル

| 項目 | 値 |
| --- | --- |
| Repository | `summerMC/Qwen3.8-27B-Longbook27-VL-GDN64-GGUF` |
| Revision | `f3ee816de60fd7f67911e949a53514b4b17f84bb` |
| Filename | `SpeedX27-GDN64-Q4_K_M.gguf` |
| Size | 16,746,122,528 bytes（約 15.60 GiB） |
| SHA-256 | `5af94bfd3e1725ab09d8b82ef929706362bc0b269a7858d893389f9a40b27521` |
| Format / quantization | GGUF v3 / Q4_K_M（file type 15、quantization version 2） |
| Model license | Apache-2.0 |

サイズと SHA-256 は [固定 revision のファイル一覧・LFS 情報][tree] に由来する。
この SHA-256 は公開された期待値であり、今回フルファイルを取得して照合した結果ではない。
機械可読な設定は [`models/longbook27.json`](../models/longbook27.json) にまとめた。

モデルの提供ファイルはこの 1 個だけで、mmproj は含まれない。[公開ファイル一覧][tree]
原本には vision の設定があるが、対象 GGUF の tensor table は言語部分だけで構成される。
そのため現在のクライアントはテキスト入力を対象とし、画像・動画・音声の処理には対応しない。
`VL` という名前だけでは画像対応の根拠にならない。

## メタデータと実体の相違

[対象 GGUF][artifact] のメタデータ 44 項目と tensor descriptor 899 個を調べた。
最初の 10,997,013 bytes が header と descriptor であり、tensor data は
32-byte alignment により offset 10,997,024 から始まる。
検査時に tensor weight data は読んでいない。

| 項目 | 宣言 / 観測値 | クライアントへの影響 |
| --- | --- | --- |
| Architecture | `qwen35` | Qwen35 の GGUF loader が必要 |
| `block_count` | 65 | 65 層すべての重みが存在するわけではない |
| `nextn_predict_layers` | 1 | MTP の宣言はあるが、対応する重みはない |
| 実際の block | `blk.0` ～ `blk.63` | main decoder は 64 層 |
| SSM conv tensor | 全 64 層に存在 | 全層 Gated DeltaNet / recurrent |
| Full attention の tensor | なし | 4 層ごとの dense attention と仮定できない |
| `full_attention_interval` | **4** | 実体と不整合な値なので topology 判定に使えない |
| recurrent layer 配列 | なし | SSM tensor の有無から topology を判定する |
| `blk.64` / nextn / MTP tensor | なし | MTP 実行を有効にしない |
| vision / projector tensor | なし | このファイルだけで vision を実行しない |

原本の [`config.json`][config] も `num_hidden_layers=64`、
全 `layer_types="linear_attention"`、`full_attention_interval=null` を宣言する。
GGUF の `full_attention_interval=4` より、実 tensor の構成を優先する必要がある。

## エンジンが必要とする修正

対象の [モデルカード][card] は冒頭で upstream llama.cpp は未対応とし、
`vnlpscale/llama.cpp` を指定している。一方、本文には「recent llama.cpp」
という一般的な説明も残っている。この不整合は、実 tensor と fork のコードで判断する。

採用元は `vnlpscale/llama.cpp` の
`c5709247c243a0b1ae299ddb9d8f4c8b0279be89`。
[`09b4a507...` の修正][fix] と [固定した Qwen35 実装][engine] が次を扱う。

1. 明示された recurrent 配列がなければ、各層の `ssm_conv1d.weight` から topology を判断する。
2. メタデータに MTP が宣言されていても、該当 tensor がなければロードをスキップする。
3. 全層 recurrent の場合に、未使用の attention KV / position graph input を作らない。

`qwen35` という architecture 名だけを確認して通常の Qwen モデルとして扱うと、
このチェックポイントの実体に合わないロード経路に入る。
エンジンの取得元・削減方針はプロジェクトの engine lock とビルド文書に従う。
モデルの Apache-2.0 とエンジンの MIT は別々に扱う。

## 必須 tensor の形状

次は GGUF の dimension 順序であり、表示順を転置していない。[GGUF descriptor][artifact]
各 `blk.N` に 14 個、global に 3 個あるので、合計は

\[
64\times14+3=899
\]

となる。

| 各 block の suffix | Dimensions |
| --- | --- |
| `attn_gate.weight` | `[5120, 6144]` |
| `attn_norm.weight` | `[5120]` |
| `attn_qkv.weight` | `[5120, 10240]` |
| `ffn_down.weight` | `[17408, 5120]` |
| `ffn_gate.weight` / `ffn_up.weight` | `[5120, 17408]` |
| `post_attention_norm.weight` | `[5120]` |
| `ssm_a` / `ssm_dt.bias` | `[48]` |
| `ssm_alpha.weight` / `ssm_beta.weight` | `[5120, 48]` |
| `ssm_conv1d.weight` | `[4, 10240]` |
| `ssm_norm.weight` | `[128]` |
| `ssm_out.weight` | `[6144, 5120]` |

Global tensor は `token_embd.weight [5120,248320]`、
`output_norm.weight [5120]`、`output.weight [5120,248320]`。
量子化の内訳は F32 が 385 個、Q4_K が 449 個、Q6_K が 65 個。
混合精度の具体的な層番号も manifest に記録している。

GDN の key head は 16 個、value head は 48 個、各 head の state dimension は 128。
`GDN64` の 64 は、このファイルでは 64 recurrent layers を表す。
head dimension が 64 という意味ではない。[原本の設定][config]

## Context とメモリ

GGUF が宣言する context length は **262,144 tokens**。
モデルカードの実行例は `-c 8192 -ngl 99` で、GPU に収まらない場合は
offload 層数を下げるとしている。[GGUF モデルカード][card]
このプロジェクトの初期 context は 8,192 tokens とし、実ハードウェアでの結果に応じて調整する。

原本の設定には native / validated が 262,144、experimental logical context が
1,000,000,000,000 とある。同時に `native_1T_context=false` と明記されている。
1T は実験的な streaming / addressable context の説明で、GGUF クライアントで
検証した context 上限ではない。[原本の設定][config]

原本モデルカードの約 55.36 GiB という peak memory は、BF16、batch 1、
chunk 8,192、約 80 GiB GPU での報告である。[原本のベンチマーク][source-card]
Q4_K_M やこのクライアントの測定値に置き換えて使わない。
ファイルサイズ約 15.60 GiB も必要 VRAM 全体ではない。
実行には recurrent state、計算用 buffer、tokenizer などの領域が加わる。

## Tokenizer とテキスト prompt

埋め込まれた template は原本の [`chat_template.jinja`][template] と同一で、
8,952 bytes、SHA-256 は
`c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`。
Tokenizer は `gpt2` / `qwen35`、vocabulary は 248,320。
特別な token は以下のとおり。

| Token | ID |
| --- | --- |
| `<\|endoftext\|>` | 248044 |
| `<\|im_start\|>` | 248045 |
| `<\|im_end\|>` | 248046 |

`add_bos_token=false`。chat の終了 token は `im_end`。
原本の generation config は `im_end` と `endoftext` の両方を EOS としている。
Sampling の値は temperature 1.0、top-k 20、top-p 0.95。[Generation 設定][generation]

テキスト会話の template は、system/user を
`<|im_start|>role\ncontent<|im_end|>\n` で囲む。
生成を開始する prefix は次のとおり。以下の `\n` は実際の LF を表す。

```text
thinking on:
<|im_start|>assistant\n<think>\n

thinking off:
<|im_start|>assistant\n<think>\n\n</think>\n\n
```

過去の assistant turn は、既定の `preserve_thinking` では以下になる。

```text
<|im_start|>assistant\n<think>\n{reasoning_content}\n</think>\n\n{content}<|im_end|>\n
```

`reasoning_content` がなければ空文字列になる。
thinking を off にして生成した turn でも、履歴の empty thinking wrapper は残る。
`preserve_thinking=false` の場合は最後の user query より前の reasoning が除外される。
本文と reasoning を分けて保持し、wrapper を二重に付けない。[原本 template][template]

原本の既定 reasoning effort は `xhigh` で、その場合は追加の system instruction を挿入する。
`medium` ではこの追加を行わない。独自の小さなテキスト renderer は、
使う effort と preservation 条件を明示してこの違いを扱う必要がある。
原本 template が持つ tool / image / video の形式は、現在のテキスト入力の対応範囲に含めない。

## ローカル検査

```bash
python3 tools/model_info.py /path/to/SpeedX27-GDN64-Q4_K_M.gguf
python3 tools/model_info.py /path/to/SpeedX27-GDN64-Q4_K_M.gguf --json
python3 tools/model_info.py /path/to/SpeedX27-GDN64-Q4_K_M.gguf --verify-sha256
```

標準ライブラリだけで動作し、自動ダウンロードやネットワーク接続は行わない。
既定では最大 32 MiB の header / descriptor を読み、metadata / array / tensor 数にも
上限を設ける。全 64 層の名前、形状、量子化、tensor offset、重複・範囲外、
tokenizer、chat template、サイズを検査する。tensor data をモデルとしてロードしない。
この限定 inspector は対象ファイルで使われる F32 / Q4_K / Q6_K のみを扱う。

検査結果は次の 2 段階を分ける。

| 結果 | 確認できること |
| --- | --- |
| `structural_match=true` / `checkpoint_identity=unverified` | 対象と同じ header・tensor 構成。重み自体の同一性は未確認 |
| `checkpoint_identity=matches_pinned_sha256` | 全ファイル SHA-256 が公開された固定値と一致 |

`--verify-sha256` のときだけ、4 MiB の buffer を使って全ファイルを順に読む。
これは 16.7 GB 分のディスク読み出しを伴う。
形式・構成がすでに不一致なら checksum を省略して失敗を返す。
同じ architecture やファイル名であっても、別の重みを対象モデルと表示しない。
SHA-256 が一致しても、推論の正しさ・性能・品質を検証したことにはならない。

構造テストは合成データによる parser / guard の検証である。
実チェックポイントの CPU/GPU ロード、生成品質、長文処理は別途実測が必要である。

### この実装で確認した範囲

2026-10-10 の検査では、固定 revision から **先頭 10,997,013 bytes だけ**を
HTTP Range で再取得し、header / descriptor とした。一時ファイルの残りは
書き込まれていない sparse 領域で、実行用モデルではない。この明示的な検査用 fixture に対し、
Python inspector と native CLI の `--inspect` がともに正常終了し、
899 tensors / 64 recurrent layers の構造を認識した。fixture は検査後に削除した。

Python の結果は `structural_match=true`、`checkpoint_identity=unverified`。
この検査で checksum、weight loading、生成は実行していない。
つまり「対象ファイルの実 header を解釈できる」ことを確認し、
「実チェックポイントで推論できる」ことは未確認のままである。

`python3 -m unittest discover -s tests -p test_model_info.py -v` の 18 tests も成功した。
異常な長さ、重複 key、範囲外 / 重複 tensor、誤った形状・量子化、
FIFO、checksum 不一致、構造一致だけでは identity を確定しないことを検査している。

[tree]: https://huggingface.co/api/models/summerMC/Qwen3.8-27B-Longbook27-VL-GDN64-GGUF/tree/f3ee816de60fd7f67911e949a53514b4b17f84bb
[artifact]: https://huggingface.co/summerMC/Qwen3.8-27B-Longbook27-VL-GDN64-GGUF/resolve/f3ee816de60fd7f67911e949a53514b4b17f84bb/SpeedX27-GDN64-Q4_K_M.gguf
[card]: https://huggingface.co/summerMC/Qwen3.8-27B-Longbook27-VL-GDN64-GGUF/raw/f3ee816de60fd7f67911e949a53514b4b17f84bb/README.md
[config]: https://huggingface.co/summerMC/Qwen3.8-27B-Longbook27-VL-GDN64/raw/a579208838a5ef31718bd031babfbf859d377a05/config.json
[source-card]: https://huggingface.co/summerMC/Qwen3.8-27B-Longbook27-VL-GDN64/raw/a579208838a5ef31718bd031babfbf859d377a05/README.md
[generation]: https://huggingface.co/summerMC/Qwen3.8-27B-Longbook27-VL-GDN64/raw/a579208838a5ef31718bd031babfbf859d377a05/generation_config.json
[template]: https://huggingface.co/summerMC/Qwen3.8-27B-Longbook27-VL-GDN64/raw/a579208838a5ef31718bd031babfbf859d377a05/chat_template.jinja
[engine]: https://github.com/vnlpscale/llama.cpp/blob/c5709247c243a0b1ae299ddb9d8f4c8b0279be89/src/models/qwen35.cpp
[fix]: https://github.com/vnlpscale/llama.cpp/commit/09b4a507062509b021ed07860fdf5dbc6727899f
