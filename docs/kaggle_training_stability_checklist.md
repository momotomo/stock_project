# Kaggle Training Stability Checklist

目的:

- GitHub Actions -> Kaggle 学習の安定性観測を毎回同じ観点で残す
- `training_run_log.csv` を当面 optional のまま扱いながら、required 化の判断材料を蓄積する

1回ごとの確認項目:

- GitHub Actions:
  - `Push Kaggle kernel` が成功
  - `Poll Kaggle kernel status` が `COMPLETE`
  - `List downloaded Kaggle output files` に `training_run_log.csv` / `run_log_written.txt` / `stock-ai-trainer.log` が出ている
- Required artifact:
  - `best_params.json`
  - `classifier_model.pkl`
  - `meta_model.pkl`
  - `ranker_model.pkl`
  - `regressor_model.pkl`
  - `scaler.pkl`
  - `selected_features.pkl`
  - `stock-ai-trainer.log`
- Optional artifact:
  - `training_run_log.csv`
  - `run_log_written.txt`
- ログ確認:
  - `RUN_LOG_PERSIST_START`
  - `RUN_LOG_PERSIST_TARGET`
  - `RUN_LOG_PERSIST_DONE exists=True`
  - `RUN_LOG_PERSIST_FAILED` が出ていない
- 判定:
  - 成功
  - warning のみで継続可能
  - 要調査

run summary テンプレート:

```text
Date:
Workflow run URL:
Kernel slug:
Push Kaggle kernel: success / failed
Poll Kaggle kernel status: COMPLETE / other
Required artifact: all present / missing
Optional artifact:
  - training_run_log.csv: present / missing
  - run_log_written.txt: present / missing
stock-ai-trainer.log:
  - RUN_LOG_PERSIST_START: yes / no
  - RUN_LOG_PERSIST_TARGET: yes / no
  - RUN_LOG_PERSIST_DONE exists=True: yes / no
  - RUN_LOG_PERSIST_FAILED: yes / no
Judgement: success / warning / investigate
Notes:
```

連続成功の記録表:

| Date | Workflow run | Kernel slug | COMPLETE | Required artifact | training_run_log.csv | run_log_written.txt | Judgement | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YYYY-MM-DD | run url | stock-ai-trainer | yes/no | ok/ng | present/missing | present/missing | success/warning/investigate | |
| YYYY-MM-DD | run url | stock-ai-trainer | yes/no | ok/ng | present/missing | present/missing | success/warning/investigate | |
| YYYY-MM-DD | run url | stock-ai-trainer | yes/no | ok/ng | present/missing | present/missing | success/warning/investigate | |

optional 維持の運用ルール:

- `training_run_log.csv` は当面 optional
- missing でも workflow を即 fail にしない
- missing の場合は warning として run summary に残す
- `run_log_written.txt` も optional として同様に扱う
- 3回連続成功、または 1 週間安定したら `training_run_log.csv` の required 化を再検討する

warning 時の最低確認:

- artifact に `stock-ai-trainer.log` が含まれているか
- `RUN_LOG_PERSIST_DONE exists=True` がログにあるか
- `List downloaded Kaggle output files` の一覧に欠けがないか
- `You cannot change the editor type of a kernel` のような editor type 問題が再発していないか
