import os
import logging
import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import joblib
from datetime import datetime
import warnings

# =========================================================
# AI予測バッチ処理 (daily_batch.py) - 暴落ブレーカー・メタラベリング搭載版
# =========================================================

warnings.simplefilter('ignore', ResourceWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 🔥 追加: 相場環境（地合い）を判定するブレーカー機能
def check_market_regime():
    """
    TOPIX (1306.T) の動きを見て、相場全体が安全か（買うべき地合いか）を判定する。
    """
    logger.info("🔍 日本市場全体（TOPIX）の地合いをチェックしています...")
    try:
        topix = yf.download("1306.T", period="1mo", progress=False)
        if topix.empty:
            logger.warning("TOPIXデータが取得できませんでした。安全のためブレーカーはパスします。")
            return True
        
        if isinstance(topix.columns, pd.MultiIndex):
            close_price = topix.xs('Close', level=0, axis=1).iloc[:, 0]
        else:
            close_price = topix['Close']
            
        latest_close = close_price.iloc[-1]
        prev_close = close_price.iloc[-2]
        sma_5 = close_price.rolling(window=5).mean().iloc[-1]
        
        pct_change = (latest_close - prev_close) / prev_close
        
        logger.info(f"📊 [相場環境] TOPIX現在値: {latest_close:.1f}, 5日SMA: {sma_5:.1f}, 前日比: {pct_change*100:.2f}%")
        
        is_safe = True
        reasons = []
        
        # ルール1: 短期ダウントレンド（5日線を下回っている）
        if latest_close < sma_5:
            is_safe = False
            reasons.append("TOPIXが5日移動平均線を下回っている(短期下落トレンド)")
        
        # ルール2: パニック急落（前日比-1.5%以上下落）
        if pct_change <= -0.015:
            is_safe = False
            reasons.append(f"TOPIXが前日比で急落している({pct_change*100:.2f}%)")
            
        if not is_safe:
            logger.warning(f"🚨 【ブレーカー発動】 地合いの悪化を検知しました: {' / '.join(reasons)}")
            
        return is_safe
        
    except Exception as e:
        logger.error(f"地合いチェック中にエラーが発生しました: {e}")
        return True # エラー時はとりあえず動かす

# --- 分析・特徴量作成関数 ---
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
        logger.warning(f"マクロデータの取得に失敗しました: {e}")
        return pd.DataFrame()

def calc_fractional_diff(series, d=0.5, window=20):
    weights = [1.0]
    for k in range(1, window):
        weights.append(-weights[-1] * (d - k + 1) / k)
    weights = np.array(weights)[::-1]
    return series.rolling(window).apply(lambda x: np.dot(weights, x), raw=True)

def get_stock_features(ticker, macro_returns):
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
    
    # 推論用なのでターゲット計算は不要ですが、欠損値処理の整合性のために残します
    data['Target_Class'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data['Target_Return'] = data['Close'].shift(-1) / data['Close'] - 1
    return data

def get_tickers():
    tickers = {}
    if os.path.exists("tickers.txt"):
        with open("tickers.txt", "r", encoding="utf-8") as f:
            for line in f:
                if ',' in line:
                    name, ticker = line.split(',', 1)
                    tickers[name.strip()] = ticker.strip()
    return tickers

def generate_signals():
    logger.info("🧠 AI予測バッチ処理(超・軽量推論モード / メタラベリング・ブレーカー対応)を開始します...")
    
    # 🔥 0. まず最初に地合い（ブレーカー）をチェック
    is_market_safe = check_market_regime()
    if not is_market_safe:
        logger.warning("🛑 本日は地合いが悪化しているため、「休むも相場」と判断し、新規の買い推論をスキップします。")
        # 推奨銘柄を空にして保存（auto_tradeに新規買いさせないため）
        pd.DataFrame().to_csv('recommendations.csv', index=False, encoding='utf-8-sig')
        logger.info("✅ recommendations.csv を空データで更新しました。（既存の建玉の監視・決済のみ行われます）")
        return # ここで処理を完全終了する

    tickers = get_tickers()
    if not tickers:
        logger.error("処理対象の銘柄がないため終了します。")
        return

    # --- 1. Kaggleで学習済みのモデルをロード ---
    try:
        ranker_model = joblib.load('models/ranker_model.pkl')
        classifier_model = joblib.load('models/classifier_model.pkl')
        meta_model = joblib.load('models/meta_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        selected_features = joblib.load('models/selected_features.pkl')
        logger.info("✅ Kaggle産 学習済みAIモデルの読み込みに成功しました！")
    except Exception as e:
        logger.error(f"❌ モデルファイルが見つかりません。ターミナルで 'git pull' を実行してKaggleからダウンロードしてください。詳細: {e}")
        return

    macro_returns = get_macro_data()
    
    # --- 2. 最新データの取得 ---
    stock_data_dict = {}
    for name, ticker in tickers.items():
        logger.info(f"[{name}] の最新データを取得中...")
        data = get_stock_features(ticker, macro_returns)
        if data is not None:
            stock_data_dict[name] = data

    if not stock_data_dict: return

    rsi_df = pd.DataFrame({name: df['RSI_14'] for name, df in stock_data_dict.items()})
    rsi_mean = rsi_df.mean(axis=1)
    rsi_std = rsi_df.std(axis=1)
    
    df_all = []
    for name in stock_data_dict.keys():
        stock_data_dict[name]['RSI_Z_Score'] = (stock_data_dict[name]['RSI_14'] - rsi_mean) / (rsi_std + 1e-9)
        stock_data_dict[name]['Ticker'] = name
        df_all.append(stock_data_dict[name])

    df_panel = pd.concat(df_all).sort_index()

    ALL_FEATURES = [
        'Log_Return_Norm', 'Frac_Diff_0.5', 'Disparity_5_Norm', 'Disparity_25_Norm', 
        'SMA_Cross', 'RSI_14', 'BB_PctB', 'BB_Bandwidth', 'MACD_Norm', 
        'ATR', 'OBV_Ret', 'USDJPY_Ret', 'SP500_Ret', 'TOPIX_Ret',
        'EMA_Diff_Ratio', 'MACD_Div', 'Log_Return_Norm_Lag1', 'Log_Return_Norm_Lag2',
        'RSI_14_Lag1', 'RSI_14_Lag2', 'MACD_Norm_Lag1', 'MACD_Norm_Lag2', 'RSI_Z_Score',
        'Excess_Return', 'Excess_Return_20d'
    ]
    
    df_panel = df_panel.dropna(subset=ALL_FEATURES)
    if df_panel.empty:
        logger.error("有効なデータがありません。")
        return

    # 今日のデータだけを抽出
    latest_date = df_panel.index.max()
    latest_df = df_panel[df_panel.index == latest_date].copy()
    
    # --- 3. データの前処理（スケーリングと特徴量選択） ---
    X_latest_raw = latest_df[ALL_FEATURES]
    X_latest_scaled = pd.DataFrame(scaler.transform(X_latest_raw), index=X_latest_raw.index, columns=ALL_FEATURES)
    X_latest_sel = X_latest_scaled[selected_features]

    # --- 4. 爆速推論 (ベースAI ＋ メタAI) ---
    logger.info("⚡ AIによる推論(メタラベリング確信度計算)を実行中...")
    latest_df['Ranking_Score'] = ranker_model.predict(X_latest_sel)
    
    base_prob = classifier_model.predict_proba(X_latest_sel)[:, 1]
    latest_df['Prob_Up'] = base_prob
    
    X_meta_latest = X_latest_sel.copy()
    X_meta_latest['Base_Prob'] = base_prob
    latest_df['Meta_Prob'] = meta_model.predict_proba(X_meta_latest)[:, 1]

    min_score = latest_df['Ranking_Score'].min()
    max_score = latest_df['Ranking_Score'].max()
    if max_score > min_score:
        latest_df['Normalized_Score'] = (latest_df['Ranking_Score'] - min_score) / (max_score - min_score) * 100
    else:
        latest_df['Normalized_Score'] = 50.0

    latest_df = latest_df.sort_values(by='Ranking_Score', ascending=False)
    
    results = []
    for i, (idx, row) in enumerate(latest_df.iterrows()):
        reasons = []
        if i == 0: reasons.append("👑 本日のAIランキング第1位銘柄です！")
        reasons.append(f"Kaggle最新AIによる相対的強さ第{i+1}位。")
        if row['Meta_Prob'] > 0.6: reasons.append("メタAIの審査もクリアし、非常に強い買い確信度を持っています！")
        
        combo_score = (row['Normalized_Score'] * 0.3) + (row['Meta_Prob'] * 100 * 0.7)

        res_dict = {
            "銘柄名": row['Ticker'],
            "今日の終値": float(row['Close']),
            "短期スコア": float(combo_score),
            "中長期スコア": float(row['Prob_Up'] * 100),
            "明日の上昇確率": float(row['Prob_Up'] * 100),
            "メタ確信度": float(row['Meta_Prob'] * 100),
            "1W 利確(>3%)": 0.0, "2W 利確(>5%)": 0.0, "1M 利確(>10%)": 0.0,
            "3M 利確(>20%)": 0.0, "6M 利確(>30%)": 0.0, "1Y 利確(>50%)": 0.0,
            "おすすめ理由": " ".join(reasons)
        }
        results.append(res_dict)

    if results:
        df_res = pd.DataFrame(results)
        df_res.to_csv('recommendations.csv', index=False, encoding='utf-8-sig')
        logger.info("✅ 推論完了！recommendations.csv を更新しました。")
    else:
        logger.warning("⚠️ 予測結果が生成されませんでした。")

if __name__ == "__main__":
    generate_signals()