# Architecture

- `daily_batch.py` はモデル読込、特徴量生成、推論、ブレーカー判定、`recommendations.csv` 出力、`daily_health_log.csv` 追記を担当します。
- `auto_trade.py` は同一ファイル内で `MarketData`、`ExecutionLogger`、`PortfolioManager`、`TradingEngine` に責務を分離しています。
- 設定は `settings_loader.py` 経由で `settings.yml` と `settings.local.yml` をマージします。後者が優先です。
- 実行ログは SIM/PROD を分離し、KPI 集計は `scripts/ops_kpi_report.py` が `logs/` に CSV/HTML を出力します。
