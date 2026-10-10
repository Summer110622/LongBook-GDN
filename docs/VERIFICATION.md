# 検証記録 — v0.1.0 開発版

実施日: 2026-10-10。以下はローカルで実行した記録であり、GitHub CI や
実 Longbook27 の品質評価の結果ではない。
GitHub CI の結果は各 PR の **Checks**、レビュー状況は当該 PR で確認する。
本実装作業では ruleset などのリポジトリ管理設定を変更していない。

## 環境

| 項目 | 値 |
| --- | --- |
| OS / architecture | Linux / x86_64 |
| Compiler | GCC 13.3.0 |
| CMake / Ninja | 4.4.4 / 1.13.2 |
| Python | 3.12.14 |
| CPU backend | SSE4.2 / AVX / AVX2 / FMA / F16C / BMI2、OpenMP off |
| GPU | 検証環境に GPU / CUDA toolkit なし |
| Engine | `vnlpscale/llama.cpp@c5709247c243a0b1ae299ddb9d8f4c8b0279be89` + 同梱 patch |

## 実施したビルドとテスト

```bash
cmake --preset cpu
cmake --build --preset cpu --parallel 4
ctest --preset cpu
```

最初のビルドで見つかった CLI help の構文誤りを修正し、CPU 実行ファイルを生成した。
その後のレビューで入力待機中の SIGTERM が入力を待ち続ける可能性を指摘したため、
POSIX signal handler を `SA_RESTART` なしで登録し、実際に stdin を開いたまま
SIGTERM を送る回帰テストを追加した。以下は修正後の結果。

| CTest | 結果 |
| --- | --- |
| `native-help` | PASS |
| `native-version` | PASS |
| `python-unit-and-native` | PASS — 27 tests、skip なし |
| `engine-provenance` | PASS |

`python-unit-and-native` は GGUF inspector の 18 tests と、実 CPU graph を動かす
native integration の 9 tests からなる。CTest 全体も 4/4 PASS。

### Native integration の確認事項

1. 合成 GGUF を metadata / tensor から全 recurrent と認識する。
2. `--allow-compatible-model` の明示なしでは合成モデルを Longbook27 と扱わず拒否する。
3. 113-token prompt の `batch=32 / ubatch=8` と `batch=128 / ubatch=128` で
   greedy の 8-token 出力が一致する。fresh context での再実行も一致する。
4. `prompt_tokens=28 + predict=4 = context=32` は成功する。
5. `29 + 4 > 32` は stdout に生成せず拒否する。
6. model metadata の context 256 を超える指定は拒否する。
7. 日本語・絵文字・空行を含む UTF-8 標準入力で全 byte / token 数を保持する。
8. `/reset` 後に同じ入力の結果が再現し、EOF で終了する。
9. 対話で `you> ` を表示した後、stdin を開いたまま SIGTERM を送ると、
   3 秒以内に終了コード 143 で終了する。stdout に生成はない。

合成モデルは `tests/make_tiny_gguf.py` が作る **316,608 bytes の random F32**。
4 recurrent blocks、宣言だけ存在する MTP 1 block、stale な interval 4 を持つ。
重みは非ゼロで、実際の loader、tokenizer、prefill、GDN graph、sampler、decode を通る。
生成前に全 logits が有限であることをクライアントが確認する。
**これを 27B Q4_K_M、GPU、長文品質の検証としては扱わない。**

### その他のローカル確認

- `--temp nan`、`--top-p 0`、`--ctx-size 4096x`、`--threads 0`、`--seed -1` が
  エラー終了コード 2 となることを確認。
- `--image` を指定すると未対応を説明して終了コード 2 を返すことを確認。
- `cmake --install` がバイナリ、クライアントとエンジンの license、追加 notices、
  engine lock、audit を同じ配布ツリーへ配置することを確認。
- Linux バイナリのリンク先は OS の libc / libstdc++ / libm / libgcc。
  別の `libllama.so` を必要としない。

## 固定エンジンの再現性

`scripts/vendor_engine.py --verify` は 515 個の保持ファイルを hash と allowlist で検査する。
この作業では、固定 archive から再生成した source と audit が byte 単位で一致すること、
保持 source の変更を拒否すること、同じサイズの破損 archive を展開前に拒否することを確認した。

| 対象 | bytes |
| --- | ---: |
| 元 archive の非圧縮通常ファイル | 166,175,683 |
| 保持 source、patch 前 | 11,045,690 |
| 保持 source、patch 後 | 11,033,319 |
| 除外した元ファイル | 155,129,993 |

これは source 容量の集計であり、実行時 memory や速度の測定ではない。
削減判断、完全な path 一覧、保持ファイル hash は `LONGBOOK27_VENDOR_AUDIT.json` にある。

## 対象 GGUF の実ヘッダー検査

対象 revision は `f3ee816de60fd7f67911e949a53514b4b17f84bb`。
HTTP Range で先頭 **10,997,013 bytes** の metadata と tensor descriptor を取得した。
tensor payload は取得していない。
本来のサイズだけを持つ一時的な sparse fixture として以下を検査した。

| 検査 | 結果 |
| --- | --- |
| Python `tools/model_info.py` | `structural_match=true` |
| Python identity | `checkpoint_identity=unverified`、SHA-256 未検査 |
| Native `--inspect` | `all_recurrent=true`、`longbook27_signature=true` |
| Native weight identity | `weights_verified=false` |

この fixture は weight loading / inference に使用せず、検査後に削除した。
native 出力のファイル範囲フィールドは、誤解を避けるため `tensor_extents_fit_file` とした。
これが true でも、内容が正しい重みであることや、ダウンロード済みであることは証明しない。
根拠となる一次資料と exact tensor profile は [MODEL.md](MODEL.md) と
`models/longbook27.json` を参照。

## 未検証の項目

| 項目 | 状態 / 次の確認 |
| --- | --- |
| 実 27B Q4_K_M の全-file SHA-256 | 未取得。利用環境で `--verify-sha256` を実施する |
| 実 27B の weight loading / 生成 | 未実施 |
| 実 27B の複数ターン / 長文 / 中断 | 未実施 |
| 262,144 context の full-length 実行 | 未実施。GGUF の宣言値であり client 保証ではない |
| CUDA / Metal | source / preset を保持。ビルドと実機推論は未実施 |
| Windows | 未実施 |
| VL / mmproj | 指定 repo に projector がなく、機能も未実装 |
| 速度 / peak RAM / peak VRAM | 実モデルでの比較は未実施 |
| GitHub CI | 各 PR の Checks で確認する。本記録のローカル PASS とは別に扱う |
| PR レビュー / ruleset | レビュー状況は各 PR で確認する。リポジトリ管理設定は本実装作業で変更していない |

これらを対応済みとしてリリースする前に、[PR ポリシー](PR_POLICY.md) の条件で
実 checkpoint・backend ごとの再現可能な記録を追加する。
