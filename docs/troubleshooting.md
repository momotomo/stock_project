# Troubleshooting

- `daily_health_log.csv` が増えない:
  - `HEALTH_LOG_PATH` の設定値を確認
  - `daily_batch.py` の実行権限と書き込み先ディレクトリを確認
- `recommendations.csv` が空:
  - ブレーカー発動の `reason` を `daily_health_log.csv` で確認
  - 停止が続く場合は `breaker_enabled`、`ret_threshold`、`cond_close_lt_ma`、`cond_ret_lt_threshold` を見て、どの条件が止めているか判断
  - 直近の傾向は `python scripts/health_log_report.py --days 14` を実行して、`health_reason_counts.csv` と `health_daily.csv` を確認
  - モデル読み込み失敗や `tickers.txt` 空も疑う
- 決済が進まない:
  - 最優先で `order_status_log*.csv` と建玉一覧を確認
  - `TIMEOUT`、`API_STATE_*`、`CANCELED/REJECTED` を確認してから新規注文ロジックを見る
- SIM と PROD のログが混ざる:
  - `trade_execution_log*.csv` / `order_status_log*.csv` のファイル名を確認
  - `IS_PRODUCTION` の設定値を再確認
- 板情報が欠落する:
  - 現在はダミー価格を使わずにスキップするため、SIM でも板欠損時は監視・新規判定が止まる
