import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import os
from datetime import datetime, timedelta

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

def run_daily_batch():
    # 日本時間の今日の日付を取得
    jst_now = datetime.utcnow() + timedelta(hours=9)
    today_str = jst_now.strftime('%Y-%m-%d')
    print(f"[{today_str}] 🚀 日次バッチ処理を開始します...")

    tickers = {
        "トヨタ自動車": "7203.T",
        "三菱UFJ": "8306.T",
        "ソニーG": "6758.T",
        "ソフトバンクG": "9984.T",
        "任天堂": "7974.T"
    }

    results = []
    
    print("🌐 マクロ経済データを取得中...")
    macro_tickers = {"USDJPY=X": "USDJPY_Ret", "^GSPC": "SP500_Ret"}
    macro_data = yf.download(list(macro_tickers.keys()), period="5y", progress=False)['Close']
    if isinstance(macro_data.columns, pd.MultiIndex):
        macro_data.columns = macro_data.columns.get_level_values(0)
    macro_returns = np.log(macro_data / macro_data.shift(1)).rename(columns=macro_tickers)

    for name, ticker in tickers.items():
        print(f"📊 {name} ({ticker}) のデータを取得・分析中...")
        stock_info = yf.Ticker(ticker).info
        per = stock_info.get('trailingPE', None) 
        
        data = yf.download(ticker, period="5y", progress=False)
        if data.empty:
            continue
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # --- 特徴量エンジニアリング ---
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
        sma_5 = data['Close'].rolling(window=5).mean()
        sma_25 = data['Close'].rolling(window=25).mean()
        data['Disparity_5'] = (data['Close'] / sma_5) - 1
        data['Disparity_25'] = (data['Close'] / sma_25) - 1
        data['SMA_Cross'] = (sma_5 / sma_25) - 1
        
        delta = data['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down
        data['RSI_14'] = 100 - (100 / (1 + rs))
        
        std_20 = data['Close'].rolling(window=20).std()
        upper_band = data['Close'].rolling(window=20).mean() + (std_20 * 2)
        lower_band = data['Close'].rolling(window=20).mean() - (std_20 * 2)
        data['BB_PctB'] = (data['Close'] - lower_band) / (upper_band - lower_band)
        data['BB_Bandwidth'] = (upper_band - lower_band) / sma_25
        
        ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD_Norm'] = (ema_12 - ema_26) / data['Close']
        
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['ATR'] = true_range.rolling(14).mean() / data['Close']
        
        obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
        data['OBV_Ret'] = obv.pct_change()
        data = data.join(macro_returns)
        
        # --- ターゲット ---
        data['Target_Class'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        data['Target_Price'] = data['Close'].shift(-1)
        
        features = ['Log_Return', 'Disparity_5', 'Disparity_25', 'SMA_Cross', 'RSI_14', 
                    'BB_PctB', 'BB_Bandwidth', 'MACD_Norm', 'ATR', 'OBV_Ret', 'USDJPY_Ret', 'SP500_Ret']
        
        data_clean_short = data.dropna(subset=features + ['Target_Class', 'Target_Price'])
        
        if len(data_clean_short) > 100:
            historical_data = data_clean_short[:-1]
            latest_data = data_clean_short.iloc[[-1]]
            current_price = latest_data['Close'].values[0]
            
            X_raw = historical_data[features]
            y_class = historical_data['Target_Class']
            y_price = historical_data['Target_Price']
            
            scaler = RobustScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), index=X_raw.index, columns=features)
            latest_X_scaled = pd.DataFrame(scaler.transform(latest_data[features]), index=latest_data.index, columns=features)
            
            tscv = TimeSeriesSplit(n_splits=5)
            accuracies = []
            for train_index, test_index in tscv.split(X_scaled):
                X_train_w, X_test_w = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
                y_train_w, y_test_w = y_class.iloc[train_index], y_class.iloc[test_index]
                eval_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                eval_model.fit(X_train_w, y_train_w)
                accuracies.append(accuracy_score(y_test_w, eval_model.predict(X_test_w)))
            wf_mean_accuracy = np.mean(accuracies)
            
            clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            clf.fit(X_scaled, y_class)
            prob_up = clf.predict_proba(latest_X_scaled)[0][1]
            
            reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            reg.fit(X_scaled, y_price)
            pred_price = reg.predict(latest_X_scaled)[0]
            
            tb_horizons = {
                '1W': {'days': 5, 'pt': 0.03, 'sl': 0.03, 'name': '1W 利確(>3%)'},
                '2W': {'days': 10, 'pt': 0.05, 'sl': 0.05, 'name': '2W 利確(>5%)'},
                '1M': {'days': 21, 'pt': 0.10, 'sl': 0.10, 'name': '1M 利確(>10%)'},
                '3M': {'days': 63, 'pt': 0.20, 'sl': 0.20, 'name': '3M 利確(>20%)'},
                '6M': {'days': 126, 'pt': 0.30, 'sl': 0.30, 'name': '6M 利確(>30%)'},
                '1Y': {'days': 252, 'pt': 0.50, 'sl': 0.50, 'name': '1Y 利確(>50%)'}
            }
            
            tb_results = {}
            for h_key, h_params in tb_horizons.items():
                target_col = f'Target_TB_{h_key}'
                data[target_col] = calc_triple_barrier(data['Close'], h_params['days'], h_params['pt'], h_params['sl'])
                valid_idx = data[data[target_col].notna()].index
                train_idx = valid_idx.intersection(X_raw.index)
                if len(train_idx) > 100:
                    clf_h = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                    clf_h.fit(X_scaled.loc[train_idx], data.loc[train_idx, target_col])
                    classes = list(clf_h.classes_)
                    prob_1 = clf_h.predict_proba(latest_X_scaled)[0][classes.index(1)] if 1 in classes else 0.0
                else:
                    prob_1 = 0.0
                tb_results[h_params['name']] = round(float(prob_1 * 100), 2)
            
            per_str = round(per, 1) if isinstance(per, (int, float)) else None
            
            result_dict = {
                "Date": today_str,
                "銘柄名": name,
                "WF平均正解率(明日)": round(float(wf_mean_accuracy * 100), 2),
                "PER(割安)": per_str,
                "今日の終値": round(float(current_price), 1),
                "明日の予測値": round(float(pred_price), 1),
                "明日の上昇確率": round(float(prob_up * 100), 2)
            }
            result_dict.update(tb_results)
            results.append(result_dict)

    if results:
        df = pd.DataFrame(results)
        csv_file = 'prediction_history.csv'
        
        # ファイルが存在しない場合は新規作成（ヘッダーあり）、存在する場合は追記（ヘッダーなし）
        if not os.path.exists(csv_file):
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            print(f"✅ 新規ファイル {csv_file} を作成し、記録しました。")
        else:
            df.to_csv(csv_file, mode='a', header=False, index=False, encoding='utf-8-sig')
            print(f"✅ 既存ファイル {csv_file} に本日のデータを追記しました。")
    else:
        print("⚠️ 記録するデータがありませんでした。")

if __name__ == "__main__":
    run_daily_batch()