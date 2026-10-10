# PR ポリシー

## 基本方針

Longbook27 は対象 GGUF に必要なコードと検証可能な互換性を維持する。
`main` は再現可能なビルドを保ち、変更は目的ごとの PR で取り込む。
初期開発では `AGENTS.md` と本ポリシーを実装前の bootstrap commit として作成した。
既存リポジトリへの導入では、その `main` の履歴を保持して PR で追加する。

## ブランチと変更単位

- `feat/<topic>`: 機能追加、`fix/<topic>`: 修正、`docs/<topic>`: 文書、
  `engine/<topic>`: エンジン更新・削減。
- 1 PR は 1 つの利用者向けの目的、または 1 つの独立したエンジン変更に絞る。
- PR タイトルは `feat: ...`、`fix: ...`、`docs: ...`、`build: ...` などで目的を示す。
- 無関係な整形、モデル重み、生成物、ビルド出力を含めない。
- 開発中は Draft、必要な検証と説明が揃ったら Ready for review とする。

## PR に必要な情報

1. 解決する問題と、変更後に利用者ができること。
2. 実装の範囲、互換性への影響、意図的に残した制約。
3. 実行したコマンドと結果。未実行の環境は明記する。
4. エンジン変更では元の commit SHA、変更後の SHA、削除したコード、
   残した依存、ライセンス通知の扱い。
5. 推論・性能を主張する場合は以下の再現条件。

| 項目 | 記録する内容 |
| --- | --- |
| モデル | repository / revision / filename / SHA-256 / 量子化 |
| エンジン | fork / commit / Longbook27 patch |
| 環境 | OS、CPU、GPU、RAM/VRAM、compiler、backend |
| 入力 | prompt または再現可能な fixture、画像なら projector |
| 設定 | context、batch/ubatch、GPU layers、seed、sampling |
| 結果 | ロード可否、生成、prefill/decode、peak memory、終了理由 |

速度の比較は同じ入力・設定・ハードウェアで行い、測定回数とばらつきを示す。
合成モデルのテストは実チェックポイントの動作確認や品質評価と区別する。

## マージ条件

- 必須 CI `cpu-build-and-test` が成功する。
- レビューの未解決指摘がなく、非著者の maintainer 1 名以上が承認する。
- 動作が変わる場合は文書・必要なテストが更新されている。
- エンジン変更は固定 SHA と再生成手順が揃い、全 recurrent layer の
  形状・state handling を壊していないことを確認する。
- 対象 GGUF の実行が未確認の変更はその旨を明示し、対応済みリリースとは扱わない。
- 通常は squash merge し、元ブランチを削除する。取り消しは revert PR で行う。

これはリポジトリ運用のポリシーであり、GitHub の保護設定を変更したという
意味ではない。リポジトリ管理者は required PR、上記 status check、承認 1 名、
stale approval の破棄、会話解決必須、force push 禁止を ruleset に設定する。
CI の実行結果は各 PR の Checks で確認し、管理設定の適用状況は別途確認する。
緊急例外が必要な場合は maintainer が理由・影響・追跡 issue を PR に記録する。

## リリース条件

CPU CI に加えて、配布対象 backend と実 GGUF の短文生成・複数ターン・
長文境界・中断処理を検証する。未検証 backend を対応済みと表示しない。
モデルの最大 context は metadata と実測に従い、巨大な論理 context の
実験値を製品保証にしない。重みのライセンスとエンジンのライセンスを分けて記録する。
