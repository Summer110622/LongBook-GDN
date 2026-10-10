# feat: add Longbook27 native client and curated GDN64 engine

LongBook-GDN への導入 PR の説明ひな形。以下の検証結果は初期実装時のローカル記録であり、
利用する際は当該 PR の変更内容・追加検証に合わせて更新する。

## 問題と変更後の動作

対象 Longbook27 GGUF は通常の Qwen35 と異なる全層 GDN 構成を持ち、
古い attention interval と存在しない MTP 層を宣言する metadata が残っている。
モデル作者の対応 fork を固定し、その補正を保持した専用クライアントを用意する。

この変更により、C++ CLI から GGUF を検査し、単発 / 対話 / 文書入力を
ローカルで処理できる。入力の文脈超過は明示的に拒否し、会話は毎ターン
fresh context に全履歴を再入力する。

## 変更範囲

- `AGENTS.md`、PR ポリシー、PR テンプレートを実装前の bootstrap commit として作成。
- 対応 fork `c5709247c243a0b1ae299ddb9d8f4c8b0279be89` から
  Qwen35 / GDN に必要な source snapshot を生成。
- 他モデル、server、Web UI、converter、common、未使用 backend を source から削減。
- archive / file hash、allowlist、patch、全ファイル audit による再現性検査。
- native CLI、厳密な metadata / checksum 検査用 Python tool、CPU CI を追加。
- CPU / CUDA / Metal preset と license / attribution 付き install を追加。

## 検証

Linux x86_64 / GCC 13.3 / CPU で configure、build、CTest が成功。
CTest は 4/4、内部 Python は 27 tests（inspector18 + native9）成功。
合成モデルは all-recurrent + absent MTP の実 graph を通し、分割 prefill の一致、
予算境界、UTF-8、reset、idle SIGTERM を確認した。
固定 GGUF の実 header / descriptor も native と Python で解釈できた。
source の byte 一致再生成と改変拒否も確認済み。
再現条件と未実施項目は `docs/VERIFICATION.md` に記載。

## 制約

実 27B 重み、CUDA / Metal / Windows、VL、最大 context の実行、実モデルの性能は未検証。
全履歴を再処理するため、長い会話では prefill cost が増える。
GitHub CI の結果は当該 PR の Checks、レビュー状況は当該 PR で確認する。
この変更には ruleset などのリポジトリ管理設定の変更を含めない。
