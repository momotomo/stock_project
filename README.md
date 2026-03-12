# stock_project

最短起動手順:

1. `cp settings.example.yml settings.local.yml`
2. `settings.local.yml` に `API_PASSWORD_SIM` / `API_PASSWORD_PROD` / `TRADE_PASSWORD` を設定
3. SIM 実行:
   - `python daily_batch.py`
   - `python auto_trade.py`
4. PROD 実行:
   - `settings.local.yml` で `IS_PRODUCTION: true`
   - `python daily_batch.py`
   - `python auto_trade.py`
5. KPI レポート:
   - `python scripts/ops_kpi_report.py --env sim --days 14`
   - `python scripts/ops_kpi_report.py --env prod --days 14`

補足:

- `settings.yml` は Git 管理用テンプレートで、ローカル上書きは `settings.local.yml` が優先されます。
- `daily_batch.py` はブレーカー停止でも `runtime/health/daily_health_log.csv` を更新します。
- runtime 生成物は `runtime/health` / `runtime/orders` / `runtime/signals` / `runtime/logs` に出力されます。
- 約定/注文状態ログは `runtime/orders/` 配下で SIM/PROD を分離します。

詳細は [docs/architecture.md](/Users/kasuyatomohiro/stock_project/docs/architecture.md)、[docs/runbook.md](/Users/kasuyatomohiro/stock_project/docs/runbook.md)、[agents.md](/Users/kasuyatomohiro/stock_project/agents.md) を参照してください。
