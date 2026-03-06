# Agents

このリポジトリの運用ルールを以下に固定します。

- `settings.yml` は Git 管理のテンプレート、秘密情報は `settings.local.yml` に置く。読み込み順は `settings.yml` -> `settings.local.yml`。
- API 接続パスワードは `API_PASSWORD_SIM` と `API_PASSWORD_PROD` を保持し、`IS_PRODUCTION` に応じて自動切替する。
- 約定ログと注文状態ログは SIM/PROD で完全分離する。
  - `trade_execution_log_SIM.csv` / `trade_execution_log.csv`
  - `order_status_log_SIM.csv` / `order_status_log.csv`
- `daily_batch.py` はブレーカー停止時でも `daily_health_log.csv` に必ず 1 行追記する。取引ゼロ日でも検証を止めない。
- 運用優先順位は「決済不成立の解消」が最優先。新規エントリー改善より先に、`order_status_log*.csv`、建玉、未約定注文の整合を確認する。
- Optuna による閾値・ハイパーパラメータ調整は週次バッチで実施し、日次実行には組み込まない。
