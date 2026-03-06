# Calibration

- エントリー閾値、ATR 係数、ブレーカー閾値は `settings.local.yml` で環境別に調整します。
- KPI の確認対象は約定遅延、スプレッド、スリッページ、TIMEOUT 率です。
- Optuna による探索は週次に限定し、日次バッチには入れません。
- ブレーカーの基本設定:
  - `BREAKER_TICKER`
  - `BREAKER_RET_THRESHOLD`
  - `BREAKER_MA_DAYS`
