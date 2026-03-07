# Kaggle Training Schedule Plan

目的:

- GitHub Actions の Kaggle 学習 workflow に schedule を追加する前提条件と運用方針を整理する
- 安定性観測フェーズの完了条件を明確にし、schedule 追加判断を後で迷わないようにする

前提:

- GitHub Actions -> Kaggle push / execute / output download は end-to-end で成功確認済み
- `training_run_log.csv` は当面 optional のまま運用する
- workflow ファイル自体にはまだ schedule を追加しない

schedule をまだ追加しない理由:

- 先に安定性観測を完了する必要がある
- `training_run_log.csv` / `run_log_written.txt` / `stock-ai-trainer.log` の出力安定性を見切れていない
- required artifact は安定しているが、optional artifact の扱いを schedule 前に固めたい

schedule を追加する前提条件:

- `Push Kaggle kernel` が成功する
- `Poll Kaggle kernel status` が `COMPLETE`
- `training_run_log.csv` / `run_log_written.txt` / `stock-ai-trainer.log` が安定して出る
- 3回連続成功、または 1 週間安定が確認できる
- `docs/kaggle_training_stability_checklist.md` の記録が埋まっている

追加候補の運用案:

- 第1候補は週次実行
- 安定観測後も、いきなり日次にはしない
- 週次で問題が無ければ、将来必要に応じて日次を再検討する

cron を決めるときの観点:

- 日本時間ベースでの実行時刻
- Kaggle 実行時間の余裕
- GitHub Actions と Kaggle の混雑を避ける時間帯
- 前後の運用確認や artifact 回収のしやすさ

失敗時の扱い:

- fail した run は checklist / summary に残す
- required artifact 欠損時は要調査
- `training_run_log.csv` 欠損は当面 warning 扱い
- `stock-ai-trainer.log` に `RUN_LOG_PERSIST_DONE exists=True` が無い場合は要調査

artifact 保管方針の候補:

- required artifact 群:
  - `best_params.json`
  - `classifier_model.pkl`
  - `meta_model.pkl`
  - `ranker_model.pkl`
  - `regressor_model.pkl`
  - `scaler.pkl`
  - `selected_features.pkl`
  - `stock-ai-trainer.log`
- optional artifact:
  - `training_run_log.csv`
  - `run_log_written.txt`
- 保持期間や保管数は schedule 追加後に見直す
- 昇格ルールが固まるまでは、artifact を監査用途で保持する前提にする

required 化との関係:

- `training_run_log.csv` の required 化を schedule 追加前または直後に無理に行わない
- 先に安定運用を確認する
- required 化は 3回連続成功または 1 週間安定の確認後に再検討する

将来の見直しポイント:

- 週次 schedule の実行曜日と時間
- failure 通知の導線
- artifact 昇格ルール
- OOS / cost_hat 観測との接続
