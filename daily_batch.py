import sys
import os
import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import joblib
from datetime import datetime
import logging
import warnings

# =========================================================
# AI予測バッチ処理 (daily_batch.py) - V2.1 完全実運用対応版
# =========================================================

warnings.simplefilter('ignore', ResourceWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ===== V2.1 Cost Model Params (ratio) =====
BASE_FEE = 0.001            # 片道手数料率（例：0.1%）
SLIPPAGE_FACTOR = 0.05      # ATRに対するスリッページ係数（初期仮置き）
TIME_LAG_PENALTY = 0.001    # 14:50決済 vs 15:00 Closeの不確実性（往復に追加）

def cost_hat_roundtrip(atr_prev_ratio: float) -> float:
    return (BASE_FEE * 2.0) + (atr_prev_ratio * SLIPPAGE_FACTOR * 2.0) + TIME_LAG_PENALTY

def check_market_breaker():
    logger.info("🔍 相場環境(地合い)の悪化をチェックしています...")
    try:
        topix = yf.download("1306.T", period="1mo", progress=False)
        if topix.empty: return False
        
        if isinstance(topix.columns, pd.MultiIndex):
            topix.columns = topix.columns.get_level_values(0)
            
        close_prices = topix['Close']
        latest_price = close_prices.iloc[-1]
        prev_price = close_prices.iloc[-2]
        ma_5 = close_prices.rolling(window=5).mean().iloc[-1]
        
        daily_return = (latest_price - prev_price) / prev_price
        
        is_downtrend = latest_price < ma_5
        is_panic = daily_return <= -0.015
        
        if is_downtrend:
            logger.warning(f"⚠️ 地合い警戒: TOPIXが5日移動平均線を下回っています (TOPIX: {latest_price:.1f}, 5日線: {ma_5:.1f})")
        if is_panic:
            logger.warning(f"🚨 パニック警戒: TOPIXが前日比 {daily_return*100:.2f}% の急落を記録しています")
            
        if is_downtrend or is_panic:
            logger.error("🛑 【ブレーカー発動】 地合いの悪化を検知しました。本日の買いシグナルは強制的に停止(0件)になります。")
            return True
            
        logger.info("✅ 地合いは正常です。バッチ処理を継続します。")
        return False
    except Exception as e:
        logger.error(f"地合いチェック中にエラーが発生しました: {e}")
        return False

def get_tickers():
    tickers = {}
    if os.path.exists("tickers.txt"):
        with open("tickers.txt", "r", encoding="utf-8") as f:
            for line in f:
                if ',' in line:
                    name, ticker = line.split(',', 1)
                    tickers[name.strip()] = ticker.strip()
    return tickers

def get_macro_data():
    macro_tickers = {"USDJPY=X": "USDJPY_Ret", "^GSPC": "SP500_Ret", "1306.T": "TOPIX_Ret"}
    try:
        macro_df = yf.download(list(macro_tickers.keys()), period="5y", progress=False)
        if isinstance(macro_df.columns, pd.MultiIndex):
            macro_data = macro_df.xs('Close', level=0, axis=1) if 'Close' in macro_df.columns.levels[0] else macro_df
        else:
            macro_data = macro_df['Close']
        macro_returns = np.log(macro_data / macro_data.shift(1)).rename(columns=macro_tickers)
        macro_returns.index = pd.to_datetime(macro_returns.index).map(lambda x: x.replace(tzinfo=None).normalize())
        return macro_returns
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
        
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Frac_Diff_0.5'] = calc_fractional_diff(data['Close'], d=0.5, window=20)
        
        tr = pd.concat([data['High']-data['Low'], abs(data['High']-data['Close'].shift()), abs(data['Low']-data['Close'].shift())], axis=1).max(axis=1)
        data['ATR'] = tr.rolling(14).mean() / data['Close']
        
        data['Disparity_5'] = (data['Close'] / data['Close'].rolling(5).mean()) - 1
        data['Disparity_25'] = (data['Close'] / data['Close'].rolling(25).mean()) - 1
        
        data['Log_Return_Norm'] = data['Log_Return'] / (data['ATR'] + 1e-9)
        data['Disparity_5_Norm'] = data['Disparity_5'] / (data['ATR'] + 1e-9)
        data['Disparity_25_Norm'] = data['Disparity_25'] / (data['ATR'] + 1e-9)
        
        data['SMA_Cross'] = (data['Close'].rolling(5).mean() / data['Close'].rolling(25).mean()) - 1
        
        delta = data['Close'].diff()
        up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
        ema_up, ema_down = up.ewm(com=13, adjust=False).mean(), down.ewm(com=13, adjust=False).mean()
        data['RSI_14'] = 100 - (100 / (1 + (ema_up / ema_down)))
        
        std_20 = data['Close'].rolling(20).std()
        ma_20 = data['Close'].rolling(20).mean()
        data['BB_PctB'] = (data['Close'] - (ma_20 - 2*std_20)) / (4*std_20 + 1e-9)
        data['BB_Bandwidth'] = (4*std_20) / (data['Close'].rolling(25).mean() + 1e-9)
        data['MACD_Norm'] = (data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()) / data['Close']
        data['OBV_Ret'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum().pct_change()

        ema_20_val = data['Close'].ewm(span=20, adjust=False).mean()
        data['EMA_Diff_Ratio'] = (data['Close'] - data['Open']) / (ema_20_val + 1e-9)

        price_slope = data['Close'].diff(5)
        macd_slope = data['MACD_Norm'].diff(5)
        data['MACD_Div'] = np.where(np.sign(price_slope) != np.sign(macd_slope), 1, 0) * macd_slope

        lag_cols = ['Log_Return_Norm', 'RSI_14', 'MACD_Norm']
        for col in lag_cols:
            data[f'{col}_Lag1'] = data[col].shift(1)
            data[f'{col}_Lag2'] = data[col].shift(2)
        
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        if not macro_returns.empty: data = data.join(macro_returns)
        for col in ['USDJPY_Ret', 'SP500_Ret', 'TOPIX_Ret']:
            if col not in data.columns: data[col] = 0.0
        data[['USDJPY_Ret', 'SP500_Ret', 'TOPIX_Ret']] = data[['USDJPY_Ret', 'SP500_Ret', 'TOPIX_Ret']].ffill().fillna(0)
        
        data['Excess_Return'] = data['Log_Return'] - data['TOPIX_Ret']
        data['Excess_Return_20d'] = data['Excess_Return'].rolling(20).sum()
        
        return data
    except Exception as e:
        logger.error(f"銘柄 {ticker} のデータ取得エラー: {e}")
        return None

def main():
    logger.info("=========================================")
    logger.info("🌅 毎朝のAI予測バッチ処理を開始します (V2.1)")
    logger.info("=========================================")

    breaker_tripped = check_market_breaker()

    try:
        ranker_model = joblib.load('models/ranker_model.pkl')
        classifier_model = joblib.load('models/classifier_model.pkl')
        meta_model = joblib.load('models/meta_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        selected_features = joblib.load('models/selected_features.pkl')
        reg_model = joblib.load('models/regressor_model.pkl')
        logger.info("✅ Kaggle産 学習済みAIモデルの読み込みに成功しました！")
    except Exception as e:
        logger.error(f"❌ モデルの読み込みに失敗: {e}")
        sys.exit(1)

    tickers = get_tickers()
    if not tickers:
        logger.error("❌ tickers.txt に銘柄が設定されていません。")
        sys.exit(1)
        
    logger.info(f"📊 {len(tickers)} 銘柄の最新データを取得中...")
    macro_returns = get_macro_data()
    stock_data_dict = {}
    
    for name, ticker in tickers.items():
        data = get_stock_features(ticker, macro_returns)
        if data is not None:
            stock_data_dict[name] = data
            
    if not stock_data_dict:
        logger.error("❌ データが取得できませんでした。")
        sys.exit(1)

    rsi_df = pd.DataFrame({name: df['RSI_14'] for name, df in stock_data_dict.items()})
    rsi_mean = rsi_df.mean(axis=1)
    rsi_std = rsi_df.std(axis=1)

    df_all = []
    for name in stock_data_dict.keys():
        stock_data_dict[name]['RSI_Z_Score'] = (stock_data_dict[name]['RSI_14'] - rsi_mean) / (rsi_std + 1e-9)
        stock_data_dict[name]['Ticker'] = name
        df_all.append(stock_data_dict[name])

    df_panel = pd.concat(df_all).sort_index()

    df_panel["ATR_Prev_Ratio"] = df_panel.groupby("Ticker")["ATR"].shift(1)

    ALL_FEATURES = [
        'Log_Return_Norm', 'Frac_Diff_0.5', 'Disparity_5_Norm', 'Disparity_25_Norm', 
        'SMA_Cross', 'RSI_14', 'BB_PctB', 'BB_Bandwidth', 'MACD_Norm', 
        'ATR', 'OBV_Ret', 'USDJPY_Ret', 'SP500_Ret', 'TOPIX_Ret',
        'EMA_Diff_Ratio', 'MACD_Div', 'Log_Return_Norm_Lag1', 'Log_Return_Norm_Lag2',
        'RSI_14_Lag1', 'RSI_14_Lag2', 'MACD_Norm_Lag1', 'MACD_Norm_Lag2', 'RSI_Z_Score',
        'Excess_Return', 'Excess_Return_20d'
    ]
    
    df_panel = df_panel.dropna(subset=ALL_FEATURES + ["ATR_Prev_Ratio"])
    if df_panel.empty:
        logger.error("❌ 前処理後にデータが空になりました。")
        sys.exit(1)

    latest_date = df_panel.index.max()
    latest_df = df_panel[df_panel.index == latest_date].copy()
    
    logger.info(f"📅 予測対象日: {latest_date.strftime('%Y-%m-%d')} のデータで推論します")

    logger.info("⚡ AIによる推論(Net Score計算)を実行中...")
    X_latest_raw = latest_df[ALL_FEATURES]
    X_latest_scaled = pd.DataFrame(scaler.transform(X_latest_raw), index=X_latest_raw.index, columns=ALL_FEATURES)
    X_latest_sel = X_latest_scaled[selected_features]
    
    latest_df['Ranking_Score'] = ranker_model.predict(X_latest_sel)
    
    base_prob = classifier_model.predict_proba(X_latest_sel)[:, 1]
    latest_df['Prob_Up'] = base_prob
    
    X_meta_latest = X_latest_sel.copy()
    X_meta_latest['Base_Prob'] = base_prob
    latest_df['Meta_Prob'] = meta_model.predict_proba(X_meta_latest)[:, 1]

    latest_df["Pred_Return"] = reg_model.predict(X_latest_sel).astype(float)
    latest_df["cost_hat"] = latest_df["ATR_Prev_Ratio"].apply(cost_hat_roundtrip).astype(float)
    latest_df["Net_Score"] = latest_df["Pred_Return"] - latest_df["cost_hat"]

    latest_df = latest_df[latest_df["Net_Score"] > 0].copy()
    latest_df = latest_df.sort_values(by="Net_Score", ascending=False)
    
    results = []
    
    if breaker_tripped:
        logger.warning("🛑 ブレーカー発動中のため、シグナルを0件で出力します。")
    else:
        for i, (idx, row) in enumerate(latest_df.iterrows()):
            reasons = []
            if i == 0:
                reasons.append("👑 Net_Score第1位（コスト控除後でも期待値が残る）")
            reasons.append(f"Pred={row['Pred_Return']*100:.2f}%, Cost={row['cost_hat']*100:.2f}% → Net={row['Net_Score']*100:.2f}%")

            ticker_code = tickers.get(row['Ticker'], "").replace(".T", "")

            res_dict = {
                "銘柄名": row["Ticker"],
                "銘柄コード": ticker_code,
                "今日の終値": float(row["Close"]),
                "Net_Score(%)": float(row["Net_Score"] * 100.0),
                "予測リターン(%)": float(row["Pred_Return"] * 100.0),
                "推定往復コスト(%)": float(row["cost_hat"] * 100.0),
                "メタ確信度": float(row["Meta_Prob"] * 100.0),
                "短期スコア": float(row["Net_Score"] * 100.0), 
                "おすすめ理由": " ".join(reasons)
            }
            results.append(res_dict)

    df_res = pd.DataFrame(results)
    
    if not df_res.empty:
        cols = ["銘柄名", "銘柄コード", "今日の終値", "Net_Score(%)", "予測リターン(%)", "推定往復コスト(%)", "メタ確信度", "短期スコア", "おすすめ理由"]
        df_res = df_res[[c for c in cols if c in df_res.columns]]
    else:
        df_res = pd.DataFrame(columns=["銘柄名", "銘柄コード", "今日の終値", "Net_Score(%)", "予測リターン(%)", "推定往復コスト(%)", "メタ確信度", "短期スコア", "おすすめ理由"])

    df_res.to_csv('recommendations.csv', index=False, encoding='utf-8-sig')
    logger.info("💾 recommendations.csv に本日の予測(Net_Score順)を保存しました。")
    
    history_file = 'prediction_history.csv'
    if not df_res.empty:
        df_hist = df_res.copy()
        df_hist.insert(0, 'Date', latest_date.strftime('%Y-%m-%d'))
        
        if os.path.exists(history_file):
            df_hist.to_csv(history_file, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df_hist.to_csv(history_file, index=False, encoding='utf-8-sig')
        logger.info("📓 prediction_history.csv に履歴を追記しました。")
    else:
        logger.info("📓 対象銘柄が0件のため、履歴の追記はスキップしました。")

    logger.info("🎉 バッチ処理がすべて完了しました！")

if __name__ == "__main__":
    main()