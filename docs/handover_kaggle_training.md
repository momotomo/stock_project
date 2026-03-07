# Kaggle Training Handover

現状:

- GitHub Actions の `Run Kaggle Notebook Manually` から Kaggle Script kernel の push / execute / output download は成功確認済み
- `kernel_slug` の既定値は `stock-ai-trainer`
- `training_run_log.csv` は `/kaggle/working/training_run_log.csv` に finally 保存する形へ修正済み
- `training_run_log.csv` は validation 上はまだ optional のまま

解決済み事項:

- `training_run_log.csv` の保存経路を finally で必ず通るようにした
- `RUN_LOG_PERSIST_START` / `RUN_LOG_PERSIST_TARGET` / `RUN_LOG_PERSIST_DONE` を `stock-ai-trainer.log` へ残すようにした
- `run_log_written.txt` を出して output 側で run log 保存確認しやすくした
- `editor type mismatch` は Script kernel 前提を runbook に明記した

成功確認済みの output:

- `best_params.json`
- `classifier_model.pkl`
- `meta_model.pkl`
- `ranker_model.pkl`
- `regressor_model.pkl`
- `scaler.pkl`
- `selected_features.pkl`
- `stock-ai-trainer.log`
- `run_log_written.txt`
- `training_run_log.csv`

今後の順番:

- runbook 更新の継続
- 安定性観測
- schedule 追加
- artifact 昇格ルール定義
- OOS / cost_hat 強化

当面の方針:

- `training_run_log.csv` は 3 回連続成功または 1 週間安定までは optional のまま
- 毎回 `downloaded files` / artifact / `stock-ai-trainer.log` を確認する
- 安定性観測には `docs/kaggle_training_stability_checklist.md` の checklist / summary テンプレートを使う
- 3 回連続成功または 1 週間安定したら required 化を再検討する
- schedule は安定性観測後に追加する
- schedule 追加前に `Push Kaggle kernel`、`COMPLETE`、`training_run_log.csv`、`run_log_written.txt`、`stock-ai-trainer.log` の安定出力を確認する
