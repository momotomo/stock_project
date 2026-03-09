# Runbook

日次手順:

1. `python daily_batch.py`
2. `recommendations.csv` と `daily_health_log.csv` を確認
3. `python auto_trade.py`
4. 取引終了後に `python scripts/ops_kpi_report.py --env sim --days 7` または `--env prod` を実行
5. ブレーカー傾向の確認が必要なら `python scripts/health_log_report.py --days 14`
6. スリッページ基礎集計は `python scripts/slippage_report.py --env sim --days 14` または `--env prod`
7. `slippage_summary.csv` / `slippage_by_time_bucket.csv` / `slippage_by_symbol.csv` で `slippage_bps` を確認
8. `python scripts/cost_hat_gap_report.py --env sim --days 14` または `--env prod` で `cost_hat` の過大/過小を確認
9. `cost_gap_bps = current_cost_hat_bps - actual_cost_bps` で、正は過大見積もり、負は過小見積もり
10. GitHub Actions から Kaggle 学習を手動起動する場合は `Run Kaggle Notebook Manually` workflow を実行
11. 必要な Secrets は `KAGGLE_USERNAME` と `KAGGLE_KEY`
12. workflow は download 後に成果物検証を行い、`best_params.json`、`classifier_model.pkl`、`meta_model.pkl`、`ranker_model.pkl`、`regressor_model.pkl`、`scaler.pkl`、`selected_features.pkl`、`stock-ai-trainer.log` が 1 つでも欠けると fail する
13. `training_run_log.csv` は現時点では validation 上 optional だが、Kaggle trainer は成功時・失敗時ともに `/kaggle/working/training_run_log.csv` を finally で保存する
14. 保存時は `stock-ai-trainer.log` に `RUN_LOG_PERSIST_START` / `RUN_LOG_PERSIST_TARGET=...` / `RUN_LOG_PERSIST_DONE exists=True/False` が出る
15. `/kaggle/working/run_log_written.txt` も作られるので、workflow ログの `kaggle-output/` 一覧と artifact で `training_run_log.csv` の有無を確認する
16. workflow には週次 schedule を追加済みで、毎週土曜 00:00 UTC / 土曜 09:00 JST に同じ workflow が走る
17. schedule run でも `Push Kaggle kernel`、`Poll Kaggle kernel status` の `COMPLETE`、`training_run_log.csv`、`run_log_written.txt`、`stock-ai-trainer.log` を確認する
18. `training_run_log.csv` は schedule run でも引き続き optional のまま扱い、required 化は別途再検討する

Kaggle 学習成功条件:

- `Push Kaggle kernel` が成功すること
- `Poll Kaggle kernel status` が `COMPLETE` まで到達すること
- `List downloaded Kaggle output files` に `training_run_log.csv` / `run_log_written.txt` / `stock-ai-trainer.log` が出ること
- required artifact の `best_params.json`、モデル群、`stock-ai-trainer.log` が揃うこと

当面の運用方針:

- `training_run_log.csv` は当面 optional のまま維持する
- missing でも即 fail にはしない
- 毎回 `downloaded files` / artifact / `stock-ai-trainer.log` を見て有無を確認する
- 3回連続成功、または 1 週間安定したら required 化を再検討する

Kaggle 側前提:

- GitHub Actions は `kaggle_ai_trainer.py` を `Script kernel` として push する
- Kaggle 側の kernel も `Script kernel` 型である必要がある
- `kernel_slug` の既定値は `stock-ai-trainer`

editor type mismatch の対処:

- エラー `You cannot change the editor type of a kernel` が出たら、既存 slug が Notebook 型で残っていないかを確認する
- `stock-ai-trainer` とは別の slug を使うか、Kaggle 側で Script 型の kernel を作り直してから再実行する
- `kernel_slug` の取り違え、過去の手動コピペ運用で残った Notebook 型 kernel の再利用を優先的に疑う

今回の障害整理:

- `training_run_log.csv` 不達は単一原因ではなく複合だった
- `finally` 到達前に保存処理が漏れる経路があった
- Kaggle 側の手動コピペ運用で GitHub 側コードと乖離していた
- GitHub Actions 側では `editor type mismatch` で `Push Kaggle kernel` が失敗するケースがあった

事前確認:

- `python auto_trade.py --dry-run`
- API 通信を行わずに、設定読込、SIM/PROD 判定、ログ出力先、主要コンポーネント初期化だけを確認できる
- `python -m unittest discover -s tests -p "test_*.py" -v`
- board / orders / positions failure、`HALT_FORCE_EXIT_UNRESOLVED`、`HALT_API_ORDER_ID_MISSING` などの安全性テストを再実行できる

`daily_health_log.csv` の主要列:

- `breaker_enabled`: 設定値の `BREAKER_ENABLED`
- `ret_threshold`: 設定値の `BREAKER_RET_THRESHOLD`
- `cond_close_lt_ma`: `topix_close < topix_ma` の判定結果
- `cond_ret_lt_threshold`: `topix_ret1 < ret_threshold` の判定結果
- `reason`: `OK` / `close_below_ma` / `ret_below_threshold` / `close_below_ma|ret_below_threshold`

`health_log_report.py`:

- `python scripts/health_log_report.py --days 14`
- `python scripts/health_log_report.py --input daily_health_log.csv --out-dir logs`
- 標準出力では対象期間、`breaker_true_rate`、`reason` 内訳、`cond_close_lt_ma_true_count`、`cond_ret_lt_threshold_true_count` を確認
- `logs/health_summary.csv`
- `logs/health_reason_counts.csv`
- `logs/health_daily.csv`
- `logs/health_summary.txt`
- `logs/health_reason_counts.html` / `logs/health_daily.html` は Plotly が使える環境で出力される

見方:

- `breaker_true_rate` が高い場合は、直近期間でブレーカー停止が常態化している
- `health_reason_counts.csv` で `close_below_ma` と `ret_below_threshold` のどちらが多いかを見る
- `cond_close_lt_ma_true_count` が多ければ MA 条件が主因
- `cond_ret_lt_threshold_true_count` が多ければ前日比条件が主因
- `both_conditions_true_count` が多ければ相場悪化が複合条件で発生している
- `health_daily.csv` / `health_daily.html` で日別の `topix_close`、`topix_ma`、`topix_ret1` と `reason` を突き合わせる

運用切替:

- SIM は `IS_PRODUCTION: false`
- PROD は `IS_PRODUCTION: true`
- API パスワードは `API_PASSWORD_SIM` / `API_PASSWORD_PROD` を同時保持できる

日次確認項目:

- `order_status_log*.csv` に `TIMEOUT` や `CANCELED/REJECTED` が増えていないか
- `daily_health_log.csv` に当日 1 行追記されているか
- ブレーカー発動日は `recommendations.csv` が空であること
- ブレーカー発動日は `breaker_event_log.csv` と `simulated_order_log.csv` を確認し、停止そのものを観測対象として扱う
- `simulated_order_log.csv` は実注文ログではなく、ブレーカーが無ければ候補になっていた entry 観測ログとして扱う
- `trade_execution_log*.csv` には `entry_or_exit` / `expected_side_price` / `slippage_pct` / `slippage_bps` / `time_bucket` / `is_force_exit` / `price_level` が追加されている
- `slippage_pct` は不利方向を正で統一し、BUY は `(actual_price-expected_ask)/expected_ask`、SELL は `(expected_bid-actual_price)/expected_bid`、`slippage_bps = slippage_pct * 10000`

マージ後チェックリスト:

ソース同期:
- `git checkout main`
- `git pull`
- 運用環境（Windows/AWS）でも最新 `main` を反映

設定ファイル:
- `settings.local.yml` が運用環境に存在する
- `settings.local.yml` は Git 管理対象外
- `API_PASSWORD_SIM` / `API_PASSWORD_PROD` / `TRADE_PASSWORD` が正しく入っている
- `settings.local.yml` が `git status` に出てこない

dry-run:
- `python auto_trade.py --dry-run` が成功する
- `IS_PRODUCTION` の値が想定通り
- 選択 API パスワードキー（SIM/PROD）が想定通り
- ログ出力先が SIM/PROD で正しく分かれている
- API 通信をしないまま正常終了する

daily_batch:
- `python daily_batch.py` が成功する
- ブレーカー停止日でも `daily_health_log.csv` が 1 行増える
- `recommendations.csv` の出力が壊れていない
- `daily_health_log.csv` の `reason` / `cond_*` / `ret_threshold` が埋まっている
- ブレーカー発動時は `breaker_event_log.csv` に `breaker_reason` / `action_taken` が追記される
- 注文候補があった日は `simulated_order_log.csv` に `blocked_by_breaker=True` / `would_open_or_close=open` が追記される

KPI:
- `python scripts/ops_kpi_report.py --env sim --days 7` が成功する
- `logs/` に CSV/HTML が出る
- 空ログでも落ちない

auto_trade（SIM）:
- kabuステ環境で `IS_PRODUCTION: false` で起動できる
- `trade_execution_log_SIM.csv` / `order_status_log_SIM.csv` にのみ出力される
- 本番ログに混ざらない

本番前の最終確認:
- `FundType` が `"  "` のまま
- 売買条件が意図せず変わっていない
- 14:50 決済後に建玉ゼロ確認できる運用のまま
