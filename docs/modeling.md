# Modeling

- `daily_batch.py` はランキング、分類、メタ分類、回帰モデルを読み込み、`Net_Score = Pred_Return - cost_hat` で候補を絞ります。
- コスト推定は `BASE_FEE`、ATR 連動スリッページ、時間ラグペナルティの和です。
- `auto_trade.py` は `runtime/signals/recommendations.csv` を読み込み、`Net_Score(%)` 優先で新規エントリー候補を並べます。
- エグジットは既存ロジックを維持しつつ、ATR 連動の初期ストップとトレールを責務分離後もそのまま使います。
