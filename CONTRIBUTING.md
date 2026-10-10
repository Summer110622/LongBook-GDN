# 開発に参加する

最初に [AGENTS.md](AGENTS.md) と [PR ポリシー](docs/PR_POLICY.md) を読む。
ローカルのビルド手順は [README](README.md)、モデルの根拠は
[MODEL.md](docs/MODEL.md)、エンジンの固定と削減手順は
[ENGINE.md](docs/ENGINE.md) に記録する。

モデル重みを取得しなくても CPU ビルドと合成 GGUF による構造テストは実行できる。
実 GGUF を使った検証と合成モデルの検証は区別して PR に記録する。
[LongBook-GDN](https://github.com/Summer110622/LongBook-GDN) の既存 `main` を基点に
目的ごとのブランチで変更し、PR を作成する。既存の履歴を保持する。

CI の成功とローカル検証の成功は別の記録である。CI の結果は各 PR の **Checks**、
レビュー状況は当該 PR で確認する。GitHub repository の ruleset は
[PR ポリシー](docs/PR_POLICY.md) に従ってリポジトリ管理者が設定する。
workflow やポリシー文書を追加しただけでは管理設定は変更されない。
