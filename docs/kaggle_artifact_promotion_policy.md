# Kaggle Artifact Promotion Policy

目的:

- Kaggle 学習成果物の artifact をどの条件で昇格候補とみなすか、docs 上で判断基準を固定する
- workflow 上の自動昇格や fail 条件にする前に、運用ルールを先に揃える

前提:

- GitHub Actions -> Kaggle push / execute / output download は end-to-end で成功確認済み
- `training_run_log.csv` は現時点では optional のまま
- schedule 追加もまだ未実施
- 今回は workflow や validation 条件は変更しない

いま厳格運用しない理由:

- まだ安定性観測フェーズであり、artifact の欠損傾向を見切れていない
- `training_run_log.csv` と `run_log_written.txt` は確認対象だが、required 化はまだ早い
- 自動昇格の条件を先に強くしすぎると、運用確認より fail 判定の調整が先行してしまう

artifact の分類:

- required artifact:
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

昇格候補とする条件:

- `Push Kaggle kernel` が成功している
- `Poll Kaggle kernel status` が `COMPLETE`
- required artifact がすべて存在する
- `stock-ai-trainer.log` が存在する
- `stock-ai-trainer.log` に保存系の異常が出ていない
- `training_run_log.csv` が存在すればより望ましい

昇格保留 / 要調査とする条件:

- required artifact 欠損
- `stock-ai-trainer.log` 欠損
- `RUN_LOG_PERSIST_FAILED` が出ている
- 学習成功でも `training_run_log.csv` 欠損なら warning または保留候補
- `run_log_written.txt` 欠損も warning として記録する

training_run_log.csv / run_log_written.txt / stock-ai-trainer.log の位置づけ:

- `stock-ai-trainer.log` は required artifact として扱う
- `training_run_log.csv` は当面 optional だが、昇格判断では確認対象に含める
- `run_log_written.txt` は run log 保存確認の補助材料として扱う
- `training_run_log.csv` と `run_log_written.txt` が揃っていれば、run log 保存の信頼度が高い

将来追加したい評価項目:

- OOS metric
- `run_id`
- `git_commit` / `code_version`
- `train_rows`
- `selected_feature_count`
- `artifact_manifest_hash`
- コスト指標や slippage 指標

運用メモ:

- まだ workflow 上の自動昇格や fail 条件にはしない
- まずは docs 上で判断基準を固定する
- `training_run_log.csv` の required 化を急がず、安定性観測の結果を優先する
- 安定運用が確認できたら、schedule 追加と合わせて昇格ルールの自動化を再検討する
