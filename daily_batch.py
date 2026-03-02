import os
import yaml
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import RobustScaler
import csv
from datetime import datetime

# =========================================================
# AI予測バッチ処理 (daily_batch.py) - 高度アーキテクチャ・外部リスト対応版
# =========================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- 分析・特徴量作成関数（app.pyと完全同期） ---
def get_macro_data():
    macro_tickers = {"USDJPY=X": "USDJPY_Ret", "^GSPC": "SP500_Ret"}
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
    
    # 基本リターン
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # 🟢 ステップ1-A: フラクショナル・ディファレンス
    data['Frac_Diff_0.5'] = calc_fractional_diff(data['Close'], d=0.5, window=20)
    
    # ATR（ボラティリティの算出）
    tr = pd.concat([data['High']-data['Low'], abs(data['High']-data['Close'].shift()), abs(data['Low']-data['Close'].shift())], axis=1).max(axis=1)
    data['ATR'] = tr.rolling(14).mean() / data['Close']
    
    # 🟢 ステップ1-B: ボラティリティによる正規化
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
    
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    if not macro_returns.empty: data = data.join(macro_returns)
    for col in ['USDJPY_Ret', 'SP500_Ret']:
        if col not in data.columns: data[col] = 0.0
    data[['USDJPY_Ret', 'SP500_Ret']] = data[['USDJPY_Ret', 'SP500_Ret']].ffill().fillna(0)
    
    # ターゲット（明日上がるか）
    data['Target_Class'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    return data

def calc_triple_barrier(prices, horizon, pt_pct, sl_pct):
    vals = prices.values
    n = len(vals)
    labels = np.full(n, np.nan)
    for i in range(n - horizon):
        p0 = vals[i]
        ub = p0 * (1 + pt_pct)
        lb = p0 * (1 - sl_pct)
        path = vals[i+1 : i+1+horizon]
        hit_ub = np.where(path >= ub)[0]
        hit_lb = np.where(path <= lb)[0]
        first_ub = hit_ub[0] if len(hit_ub) > 0 else horizon + 1
        first_lb = hit_lb[0] if len(hit_lb) > 0 else horizon + 1
        if first_ub == (horizon + 1) and first_lb == (horizon + 1):
            labels[i] = 0
        elif first_ub < first_lb:
            labels[i] = 1
        elif first_lb < first_ub:
            labels[i] = -1
        else:
            labels[i] = 0
    return labels

def get_tickers():
    # 🔥 修正: ハードコーディングを廃止し、外部ファイル(tickers.txt)からのみ読み込む
    tickers = {}
    if os.path.exists("tickers.txt"):
        with open("tickers.txt", "r", encoding="utf-8") as f:
            for line in f:
                if ',' in line:
                    name, ticker = line.split(',', 1)
                    tickers[name.strip()] = ticker.strip()
    else:
        logger.error("tickers.txt が見つからないため、分析をスキップします。")
    return tickers

def generate_signals():
    logger.info("🧠 AI予測バッチ処理(高度化モデル)を開始します...")
    
    tickers = get_tickers()
    if not tickers:
        logger.error("処理対象の銘柄がないため終了します。")
        return

    macro_returns = get_macro_data()
    
    tb_horizons = {'1W': 5, '2W': 10, '1M': 21, '3M': 63, '6M': 126, '1Y': 252}
    pt_sl = {'1W': 0.03, '2W': 0.05, '1M': 0.10, '3M': 0.20, '6M': 0.30, '1Y': 0.50}
    
    # 🔥 AIの視力矯正（正規化・定常化済み）特徴量
    features = [
        'Log_Return_Norm', 'Frac_Diff_0.5', 'Disparity_5_Norm', 'Disparity_25_Norm', 
        'SMA_Cross', 'RSI_14', 'BB_PctB', 'BB_Bandwidth', 'MACD_Norm', 
        'ATR', 'OBV_Ret', 'USDJPY_Ret', 'SP500_Ret'
    ]

    results = []

    for name, ticker in tickers.items():
        logger.info(f"[{name}] のデータを取得・分析中...")
        data = get_stock_features(ticker, macro_returns)
        if data is None: continue
        
        data_features = data.dropna(subset=features)
        if len(data_features) <= 100: continue
        
        latest_data = data_features.iloc[[-1]]
        current_price = latest_data['Close'].values[0]
        historical_data = data_features.dropna(subset=['Target_Class'])
        
        if len(historical_data) <= 100: continue
        
        X_raw = historical_data[features]
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), index=X_raw.index, columns=features)
        latest_X_scaled = pd.DataFrame(scaler.transform(latest_data[features]), index=latest_data.index, columns=features)
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        
        # 1. 明日の上昇確率 と メタラベリング
        y_train_tom = historical_data['Target_Class']
        if len(np.unique(y_train_tom)) > 1:
            # 【1人目のAI】上がるか下がるかの方向性を予測
            clf.fit(X_scaled, y_train_tom)
            prob_up = clf.predict_proba(latest_X_scaled)[0][1]
            
            # 🔥 最終ステップ: メタラベリング（交差検証による確信度の予測）
            # 過去のデータで「1人目のAIの予測が当たったか(1)外れたか(0)」を判定し、それを学習ラベルとする
            oos_preds = cross_val_predict(clf, X_scaled, y_train_tom, cv=4)
            meta_labels = (oos_preds == y_train_tom).astype(int)
            
            # 【2人目のAI】1人目の予測が当たる「確信度」を予測
            clf_meta = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            clf_meta.fit(X_scaled, meta_labels)
            meta_confidence = clf_meta.predict_proba(latest_X_scaled)[0][1]
        else:
            prob_up = 0.0
            meta_confidence = 0.0
            
        # 2. 全スパンのトリプルバリア予測
        tb_results = {}
        for h_key, h_days in tb_horizons.items():
            target_col = f'Target_TB_{h_key}'
            data[target_col] = calc_triple_barrier(data['Close'], h_days, pt_sl[h_key], pt_sl[h_key])
            valid_idx = data[data[target_col].notna()].index.intersection(X_raw.index)
            if len(valid_idx) > 100:
                y_train_tb = data.loc[valid_idx, target_col]
                if len(np.unique(y_train_tb)) > 1:
                    clf.fit(X_scaled.loc[valid_idx], y_train_tb)
                    cls = list(clf.classes_)
                    tb_results[h_key] = clf.predict_proba(latest_X_scaled)[0][cls.index(1)] if 1 in cls else 0.0
                else: tb_results[h_key] = 0.0
            else: tb_results[h_key] = 0.0
            
        # 🔥 短期スコア・中長期スコアの算出（加重平均）
        short_score = (prob_up * 0.4) + (tb_results.get('1W', 0) * 0.3) + (tb_results.get('2W', 0) * 0.2) + (tb_results.get('1M', 0) * 0.1)
        long_score = (tb_results.get('1M', 0) * 0.2) + (tb_results.get('3M', 0) * 0.3) + (tb_results.get('6M', 0) * 0.3) + (tb_results.get('1Y', 0) * 0.2)
        
        # おすすめ理由の生成
        reasons = []
        if short_score > 0.6: reasons.append("短期的な上昇モメンタムが非常に強いです。")
        elif short_score > 0.55: reasons.append("短期的なテクニカル指標が好転しています。")
        
        if long_score > 0.6: reasons.append("中長期でのトレンド転換・上昇サインが点灯しています。")
        elif long_score > 0.55: reasons.append("中長期的な下値支持線からの反発が見込めます。")
        
        if data_features['RSI_14'].iloc[-1] < 30: reasons.append("RSIが売られすぎ水準にあり、反発が期待できます。")
        elif data_features['RSI_14'].iloc[-1] > 70: reasons.append("高値圏にあり、利益確定売りに注意が必要です。")
        
        if len(reasons) == 0: reasons.append("目立ったテクニカル指標の偏りはありません。")

        # 結果を辞書にまとめる
        res_dict = {
            "銘柄名": name,
            "今日の終値": float(current_price),
            "短期スコア": float(short_score * 100),
            "中長期スコア": float(long_score * 100),
            "明日の上昇確率": float(prob_up * 100),
            "メタ確信度": float(meta_confidence * 100),
            "1W 利確(>3%)": float(tb_results.get('1W', 0) * 100),
            "2W 利確(>5%)": float(tb_results.get('2W', 0) * 100),
            "1M 利確(>10%)": float(tb_results.get('1M', 0) * 100),
            "3M 利確(>20%)": float(tb_results.get('3M', 0) * 100),
            "6M 利確(>30%)": float(tb_results.get('6M', 0) * 100),
            "1Y 利確(>50%)": float(tb_results.get('1Y', 0) * 100),
            "おすすめ理由": " ".join(reasons)
        }
        results.append(res_dict)

    if results:
        # CSVファイルへ出力
        df_res = pd.DataFrame(results)
        df_res.to_csv('recommendations.csv', index=False, encoding='utf-8-sig')
        logger.info("✅ 本日のAI予測バッチが完了し、recommendations.csv を更新しました。")
    else:
        logger.warning("⚠️ 予測結果が生成されませんでした。")

if __name__ == "__main__":
    generate_signals()