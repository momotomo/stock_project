import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from datetime import datetime
import warnings

# =========================================================
# AI予測バッチ処理 (daily_batch.py) - V2.1 完全実運用対応版
# =========================================================

warnings.simplefilter('ignore', ResourceWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ===== V2.1 Cost Model Params (ratio) =====
BASE_FEE = 0.001            # 片道手数料率（例：0.1%）
SLIPPAGE_FACTOR = 0.05      # ATRに対するスリッページ係数
TIME_LAG_PENALTY = 0.001    # 14:50決済 vs 15:00 Close の不確実性

def cost_hat_roundtrip(atr_prev_ratio: float) -> float:
    # すべて比率で返す（2%なら0.02）
    return (BASE_FEE * 2.0) + (atr_prev_ratio * SLIPPAGE_FACTOR * 2.0) + TIME_LAG_PENALTY

def get_tickers():
    tickers = {}
    if os.path.exists("tickers.txt"):
        with open("tickers.txt", "r", encoding="utf-8") as f:
            for line in f:
                if "," in line:
                    name, tk = line.split(",", 1)
                    tickers[name.strip()] = tk.strip()
    return tickers

def get_macro_data():
    macro_tickers = {"USDJPY=X": "USDJPY_Ret", "^GSPC": "SP500_Ret", "1306.T": "TOPIX_Ret"}
    try:
        macro_df = yf.download(list(macro_tickers.keys()), period="5y", progress=False)
        macro_close = macro_df.xs("Close", level=0, axis=1) if isinstance(macro_df.columns, pd.MultiIndex) else macro_df["Close"]
        macro_ret = np.log(macro_close / macro_close.shift(1)).rename(columns=macro_tickers)
        macro_ret.index = pd.to_datetime(macro_ret.index).map(lambda x: x.replace(tzinfo=None).normalize())
        return macro_ret
    except Exception as e:
        logger.error(f"マクロデータの取得エラー: {e}")
        return pd.DataFrame()

def calc_fractional_diff(series, d=0.5, window=20):
    weights = [1.0]
    for k in range(1, window):
        weights.append(-weights[-1] * (d - k + 1) / k)
    weights = np.array(weights)[::-1]
    return series.rolling(window).apply(lambda x: np.dot(weights, x), raw=True)

def get_stock_features(ticker, macro_returns):
    try:
        data = yf.download(ticker, period="5y", progress=False)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.loc[:, ~data.columns.duplicated()].copy()
        data.index = pd.to_datetime(data.index).map(lambda x: x.replace(tzinfo=None).normalize())

        data["Log_Return"] = np.log(data["Close"] / data["Close"].shift(1))
        data["Frac_Diff_0.5"] = calc_fractional_diff(data["Close"], d=0.5, window=20)

        tr = pd.concat([
            data["High"] - data["Low"],
            (data["High"] - data["Close"].shift()).abs(),
            (data["Low"] - data["Close"].shift()).abs()
        ], axis=1).max(axis=1)
        data["ATR"] = tr.rolling(14).mean() / data["Close"]   # 比率ATR

        data["Disparity_5"] = (data["Close"] / data["Close"].rolling(5).mean()) - 1
        data["Disparity_25"] = (data["Close"] / data["Close"].rolling(25).mean()) - 1
        data["Log_Return_Norm"] = data["Log_Return"] / (data["ATR"] + 1e-9)
        data["Disparity_5_Norm"] = data["Disparity_5"] / (data["ATR"] + 1e-9)
        data["Disparity_25_Norm"] = data["Disparity_25"] / (data["ATR"] + 1e-9)

        data["SMA_Cross"] = (data["Close"].rolling(5).mean() / data["Close"].rolling(25).mean()) - 1

        delta = data["Close"].diff()
        ema_up = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        ema_down = (-1 * delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        data["RSI_14"] = 100 - (100 / (1 + (ema_up / (ema_down + 1e-9))))

        std_20 = data["Close"].rolling(20).std()
        ma_20 = data["Close"].rolling(20).mean()
        data["BB_PctB"] = (data["Close"] - (ma_20 - 2 * std_20)) / (4 * std_20 + 1e-9)
        data["BB_Bandwidth"] = (4 * std_20) / (data["Close"].rolling(25).mean() + 1e-9)

        data["MACD_Norm"] = (data["Close"].ewm(span=12).mean() - data["Close"].ewm(span=26).mean()) / data["Close"]
        data["OBV_Ret"] = (np.sign(data["Close"].diff()) * data["Volume"]).fillna(0).cumsum().pct_change()

        ema_20 = data["Close"].ewm(span=20, adjust=False).mean()
        data["EMA_Diff_Ratio"] = (data["Close"] - data["Open"]) / (ema_20 + 1e-9)

        price_slope = data["Close"].diff(5)
        macd_slope = data["MACD_Norm"].diff(5)
        data["MACD_Div"] = np.where(np.sign(price_slope) != np.sign(macd_slope), 1, 0) * macd_slope

        for col in ["Log_Return_Norm", "RSI_14", "MACD_Norm"]:
            data[f"{col}_Lag1"] = data[col].shift(1)
            data[f"{col}_Lag2"] = data[col].shift(2)

        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        if not macro_returns.empty:
            data = data.join(macro_returns)
        for col in ["USDJPY_Ret", "SP500_Ret", "TOPIX_Ret"]:
            if col not in data.columns:
                data[col] = 0.0
        data[["USDJPY_Ret", "SP500_Ret", "TOPIX_Ret"]] = data[["USDJPY_Ret", "SP500_Ret", "TOPIX_Ret"]].ffill().fillna(0)

        data["Excess_Return"] = data["Log_Return"] - data["TOPIX_Ret"]
        data["Excess_Return_20d"] = data["Excess_Return"].rolling(20).sum()

        return data
    except Exception as e:
        logger.error(f"銘柄 {ticker} のデータ取得エラー: {e}")
        return None

def check_market_breaker():
    logger.info("🔍 相場環境(地合い)の悪化をチェックしています...")
    try:
        df = yf.download("1306.T", period="60d", progress=False)
        if df.empty: return False
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        close = float(df["Close"].iloc[-1])
        ret1 = float(df["Close"].pct_change().iloc[-1])
        ma5 = float(df["Close"].rolling(5).mean().iloc[-1])
        
        is_downtrend = close < ma5
        is_panic = ret1 <= -0.015
        
        if is_downtrend: logger.warning(f"⚠️ 地合い警戒: TOPIXが5日線を下回っています")
        if is_panic: logger.warning(f"🚨 パニック警戒: TOPIXが前日比 {ret1*100:.2f}% 急落")
            
        if is_downtrend or is_panic:
            logger.error("🛑 【ブレーカー発動】 本日の買いシグナルは強制的に停止(0件)になります。")
            return True
            
        return False
    except Exception as e:
        logger.error(f"地合いチェックエラー: {e}")
        return False

def main():
    logger.info("=========================================")
    logger.info("🌅 毎朝のAI予測バッチ処理を開始します (V2.1 Net_Score)")
    logger.info("=========================================")

    if check_market_breaker():
        logger.warning("🛑 ブレーカー発動：recommendations.csv を空で出力します")
        pd.DataFrame(columns=["銘柄名", "銘柄コード", "今日の終値", "Net_Score(%)", "予測リターン(%)", "推定往復コスト(%)", "メタ確信度", "短期スコア", "おすすめ理由"]).to_csv("recommendations.csv", index=False, encoding="utf-8-sig")
        return

    # --- models load ---
    try:
        ranker_model = joblib.load("models/ranker_model.pkl")
        classifier_model = joblib.load("models/classifier_model.pkl")
        meta_model = joblib.load("models/meta_model.pkl")
        reg_model = joblib.load("models/regressor_model.pkl")  # ★必須
        scaler = joblib.load("models/scaler.pkl")
        selected_features = joblib.load("models/selected_features.pkl")
        logger.info("✅ Kaggle産 学習済みAIモデルの読み込みに成功しました！")
    except Exception as e:
        logger.error(f"❌ モデルの読み込みに失敗: {e}")
        sys.exit(1)

    tickers = get_tickers()
    if not tickers:
        logger.error("❌ tickers.txt が空です")
        sys.exit(1)

    logger.info(f"📊 {len(tickers)} 銘柄の最新データを取得中...")
    macro = get_macro_data()

    # --- data build ---
    stock_dict = {}
    for name, tk in tickers.items():
        df = get_stock_features(tk, macro)
        if df is not None and not df.empty:
            df["Ticker"] = name
            df["TickerCode"] = tk.replace(".T", "")
            stock_dict[name] = df

    if not stock_dict:
        logger.error("❌ データ取得失敗")
        sys.exit(1)

    rsi_df = pd.DataFrame({k: v["RSI_14"] for k, v in stock_dict.items()})
    rsi_mean, rsi_std = rsi_df.mean(axis=1), rsi_df.std(axis=1)

    df_all = []
    for name, df in stock_dict.items():
        df = df.copy()
        df["RSI_Z_Score"] = (df["RSI_14"] - rsi_mean) / (rsi_std + 1e-9)
        df_all.append(df)

    panel = pd.concat(df_all).sort_index()

    # ATR_Prev（未来リーク排除）
    panel["ATR_Prev_Ratio"] = panel.groupby("Ticker")["ATR"].shift(1)

    ALL_FEATURES = [
        "Log_Return_Norm","Frac_Diff_0.5","Disparity_5_Norm","Disparity_25_Norm",
        "SMA_Cross","RSI_14","BB_PctB","BB_Bandwidth","MACD_Norm",
        "ATR","OBV_Ret","USDJPY_Ret","SP500_Ret","TOPIX_Ret",
        "EMA_Diff_Ratio","MACD_Div","Log_Return_Norm_Lag1","Log_Return_Norm_Lag2",
        "RSI_14_Lag1","RSI_14_Lag2","MACD_Norm_Lag1","MACD_Norm_Lag2","RSI_Z_Score",
        "Excess_Return","Excess_Return_20d",
    ]

    panel = panel.dropna(subset=ALL_FEATURES + ["ATR_Prev_Ratio"])
    if panel.empty:
        logger.error("❌ 前処理後データが空")
        sys.exit(1)

    latest_date = panel.index.max()
    latest = panel.loc[panel.index == latest_date].copy()
    
    logger.info(f"📅 予測対象日: {latest_date.strftime('%Y-%m-%d')} のデータで推論します")

    X_raw = latest[ALL_FEATURES]
    X_scaled = pd.DataFrame(scaler.transform(X_raw), index=X_raw.index, columns=ALL_FEATURES)
    X_sel = X_scaled[selected_features]

    logger.info("⚡ AIによる推論(Net Score計算)を実行中...")
    # --- predict ---
    latest["Ranking_Score"] = ranker_model.predict(X_sel)
    base_prob = classifier_model.predict_proba(X_sel)[:, 1]
    latest["Prob_Up"] = base_prob

    X_meta = X_sel.copy()
    X_meta["Base_Prob"] = base_prob
    latest["Meta_Prob"] = meta_model.predict_proba(X_meta)[:, 1]

    latest["Pred_Return"] = reg_model.predict(X_sel)  # 比率（例：0.01 = +1%）

    # --- cost / net ---
    latest["cost_hat"] = latest["ATR_Prev_Ratio"].apply(cost_hat_roundtrip)
    latest["Net_Score"] = latest["Pred_Return"] - latest["cost_hat"]  # 比率

    # フィルタ（最低限：ネット期待値がプラス）
    latest = latest[latest["Net_Score"] > 0].copy()

    if latest.empty:
        logger.info("😴 本日は Net_Score>0 の銘柄がありませんでした")
        pd.DataFrame(columns=["銘柄名", "銘柄コード", "今日の終値", "Net_Score(%)", "予測リターン(%)", "推定往復コスト(%)", "メタ確信度", "短期スコア", "おすすめ理由"]).to_csv("recommendations.csv", index=False, encoding="utf-8-sig")
        return

    latest = latest.sort_values("Net_Score", ascending=False)

    results = []
    for i, (idx, row) in enumerate(latest.iterrows()):
        reasons = []
        if i == 0:
            reasons.append("👑 Net_Score第1位（コスト控除後でも期待値が残る）")
        reasons.append(f"Pred={row['Pred_Return']*100:.2f}%, Cost={row['cost_hat']*100:.2f}% → Net={row['Net_Score']*100:.2f}%")

        res_dict = {
            "銘柄名": row["Ticker"],
            "銘柄コード": row["TickerCode"],
            "今日の終値": float(row["Close"]),
            "Net_Score(%)": float(row["Net_Score"] * 100.0),
            "予測リターン(%)": float(row["Pred_Return"] * 100.0),
            "推定往復コスト(%)": float(row["cost_hat"] * 100.0),
            "メタ確信度": float(row["Meta_Prob"] * 100.0),
            "短期スコア": float(row["Net_Score"] * 100.0), # auto_tradeとの互換性維持
            "おすすめ理由": " ".join(reasons)
        }
        results.append(res_dict)

    df_res = pd.DataFrame(results)

    cols = ["銘柄名", "銘柄コード", "今日の終値", "Net_Score(%)", "予測リターン(%)", "推定往復コスト(%)", "メタ確信度", "短期スコア", "おすすめ理由"]
    df_res = df_res[[c for c in cols if c in df_res.columns]]

    df_res.to_csv('recommendations.csv', index=False, encoding='utf-8-sig')
    logger.info("✅ V2.1 recommendations.csv を更新しました（Net_Score採用）")
    
    # 履歴への追記
    history_file = 'prediction_history.csv'
    df_hist = df_res.copy()
    df_hist.insert(0, 'Date', latest_date.strftime('%Y-%m-%d'))
    
    if os.path.exists(history_file):
        df_hist.to_csv(history_file, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        df_hist.to_csv(history_file, index=False, encoding='utf-8-sig')
    logger.info("📓 prediction_history.csv に履歴を追記しました。")

if __name__ == "__main__":
    main()