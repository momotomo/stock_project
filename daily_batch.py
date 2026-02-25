import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
import os
import warnings
from datetime import datetime, timedelta

# --- 警告の非表示設定 ---
# yfinance等のライブラリ内部から発生する無害なリソース警告を非表示にする
warnings.simplefilter('ignore', ResourceWarning)

# --- トリプルバリア法のラベリング関数 ---
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

# フルセットの特徴量定義
FEATURES = [
    'Log_Return', 'Log_Return_Lag1', 'Log_Return_Lag2', 
    'Ret_RollStd_20',                                   
    'Disparity_5', 'Disparity_25', 'SMA_Cross', 
    'RSI_14', 'MFI_14',                                 
    'BB_PctB', 'BB_Bandwidth', 'MACD_Norm', 'ATR', 'OBV_Ret', 
    'USDJPY_Ret', 'SP500_Ret', 'VIX_Ret', 'US10Y_Ret'   
]

def run_daily_batch():
    jst_now = datetime.utcnow() + timedelta(hours=9)
    today_str = jst_now.strftime('%Y-%m-%d')
    print(f"[{today_str}] 🚀 日次バッチ処理（全期間・スタッキング版）を開始します...")

    tickers = {}
    if os.path.exists("tickers.txt"):
        with open("tickers.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if ',' in line:
                    name, ticker = line.split(',', 1)
                    tickers[name.strip()] = ticker.strip()
    
    if not tickers:
        tickers = {"トヨタ自動車": "7203.T", "三菱UFJ": "8306.T", "ソニーG": "6758.T"}

    all_results = []
    
    # --- マクロ経済データの取得 ---
    print("🌐 マクロ経済データ(為替, SP500, VIX, 金利)を取得中...")
    macro_tickers = {
        "USDJPY=X": "USDJPY_Ret", "^GSPC": "SP500_Ret",
        "^VIX": "VIX_Ret", "^TNX": "US10Y_Ret"
    }
    try:
        macro_df = yf.download(list(macro_tickers.keys()), period="5y", progress=False)
        if isinstance(macro_df.columns, pd.MultiIndex):
            macro_data = macro_df.xs('Close', level=0, axis=1) if 'Close' in macro_df.columns.levels[0] else macro_df
        else:
            macro_data = macro_df['Close']
        macro_returns = np.log(macro_data / macro_data.shift(1)).rename(columns=macro_tickers)
        macro_returns.index = pd.to_datetime(macro_returns.index).map(lambda x: x.replace(tzinfo=None).normalize())
    except Exception as e:
        print(f"⚠️ マクロ指標の取得に失敗: {e}")
        macro_returns = pd.DataFrame()

    for name, ticker in tickers.items():
        print(f"📊 {name} ({ticker}) の高度な分析を開始...")
        try:
            data = yf.download(ticker, period="5y", progress=False)
            if data.empty: continue
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            data = data.loc[:, ~data.columns.duplicated()].copy()
            data.index = pd.to_datetime(data.index).map(lambda x: x.replace(tzinfo=None).normalize())
            
            # --- 特徴量エンジニアリング ---
            data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
            data['Disparity_5'] = (data['Close'] / data['Close'].rolling(5).mean()) - 1
            data['Disparity_25'] = (data['Close'] / data['Close'].rolling(25).mean()) - 1
            data['SMA_Cross'] = (data['Close'].rolling(5).mean() / data['Close'].rolling(25).mean()) - 1
            
            delta = data['Close'].diff()
            up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
            ema_up, ema_down = up.ewm(com=13, adjust=False).mean(), down.ewm(com=13, adjust=False).mean()
            data['RSI_14'] = 100 - (100 / (1 + (ema_up / ema_down)))
            
            tp = (data['High'] + data['Low'] + data['Close']) / 3
            rmf = tp * data['Volume']
            pos_mf = pd.Series(np.where(tp > tp.shift(1), rmf, 0), index=data.index)
            neg_mf = pd.Series(np.where(tp < tp.shift(1), rmf, 0), index=data.index)
            mfi_ratio = pos_mf.rolling(14).sum() / (neg_mf.rolling(14).sum() + 1e-9)
            data['MFI_14'] = 100 - (100 / (1 + mfi_ratio))
            
            data['Log_Return_Lag1'] = data['Log_Return'].shift(1)
            data['Log_Return_Lag2'] = data['Log_Return'].shift(2)
            data['Ret_RollStd_20'] = data['Log_Return'].rolling(20).std()
            
            std_20, ma_20 = data['Close'].rolling(20).std(), data['Close'].rolling(20).mean()
            data['BB_PctB'] = (data['Close'] - (ma_20 - 2*std_20)) / (4*std_20 + 1e-9)
            data['BB_Bandwidth'] = (4*std_20) / (data['Close'].rolling(25).mean() + 1e-9)
            data['MACD_Norm'] = (data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()) / data['Close']
            
            tr = pd.concat([data['High']-data['Low'], abs(data['High']-data['Close'].shift()), abs(data['Low']-data['Close'].shift())], axis=1).max(axis=1)
            data['ATR'] = tr.rolling(14).mean() / data['Close']
            data['OBV_Ret'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum().pct_change()
            
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            if not macro_returns.empty: data = data.join(macro_returns)
            for col in ['USDJPY_Ret', 'SP500_Ret', 'VIX_Ret', 'US10Y_Ret']:
                if col not in data.columns: data[col] = 0.0
            data[['USDJPY_Ret', 'SP500_Ret', 'VIX_Ret', 'US10Y_Ret']] = data[['USDJPY_Ret', 'SP500_Ret', 'VIX_Ret', 'US10Y_Ret']].ffill().fillna(0)
            
            # 明日のターゲット
            data['Target_Class'] = (data['Close'].shift(-1) > data['Close']).astype(int)
            data['Target_Price'] = data['Close'].shift(-1)
            
            data_features = data.dropna(subset=FEATURES)
            if len(data_features) < 100: continue
            
            latest_data = data_features.iloc[[-1]]
            current_price = latest_data['Close'].values[0]
            historical_data = data_features.dropna(subset=['Target_Class', 'Target_Price'])
            if len(historical_data) < 100: continue
            
            X_raw = historical_data[FEATURES]
            scaler = RobustScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), index=X_raw.index, columns=FEATURES)
            latest_X_scaled = pd.DataFrame(scaler.transform(latest_data[FEATURES]), index=latest_data.index, columns=FEATURES)
            
            base_models = [
                ("RF", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
                ("GB", HistGradientBoostingClassifier(max_depth=5, random_state=42))
            ]
            
            # 1. 明日の上昇確率
            y_train_tom = historical_data['Target_Class']
            if len(np.unique(y_train_tom)) > 1:
                clf_stack_tom = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(random_state=42), cv=3)
                clf_stack_tom.fit(X_scaled, y_train_tom)
                prob_up = clf_stack_tom.predict_proba(latest_X_scaled)[0][1]
            else:
                prob_up = 0.0
            
            # 2. 予測価格
            reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            reg.fit(X_scaled, historical_data['Target_Price'])
            pred_price = reg.predict(latest_X_scaled)[0]
            
            # --- 3. 複数のスパンでのトリプルバリア予測 ---
            tb_horizons = {'1W': 5, '2W': 10, '1M': 21, '3M': 63, '6M': 126, '1Y': 252}
            pt_sl = {'1W': 0.03, '2W': 0.05, '1M': 0.10, '3M': 0.20, '6M': 0.30, '1Y': 0.50}
            tb_results = {}
            
            for h_key, h_days in tb_horizons.items():
                target_col = f'Target_TB_{h_key}'
                data[target_col] = calc_triple_barrier(data['Close'], h_days, pt_sl[h_key], pt_sl[h_key])
                valid_idx = data[data[target_col].notna()].index.intersection(X_raw.index)
                
                # 計算コストを下げるため、有効なデータが揃っている場合のみStackingを実行
                if len(valid_idx) > 100:
                    y_train_tb = data.loc[valid_idx, target_col]
                    # 💡 エラー対策: クラス（0, 1, -1）が2種類以上存在するかチェック
                    if len(np.unique(y_train_tb)) > 1:
                        clf_stack_h = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(random_state=42), cv=3)
                        clf_stack_h.fit(X_scaled.loc[valid_idx], y_train_tb)
                        cls = list(clf_stack_h.classes_)
                        p1 = clf_stack_h.predict_proba(latest_X_scaled)[0][cls.index(1)] if 1 in cls else 0.0
                        tb_results[f"{h_key} 利確(>{int(pt_sl[h_key]*100)}%)"] = round(float(p1 * 100), 2)
                    else:
                        tb_results[f"{h_key} 利確(>{int(pt_sl[h_key]*100)}%)"] = 0.0
                else:
                    tb_results[f"{h_key} 利確(>{int(pt_sl[h_key]*100)}%)"] = 0.0

            # 結果格納
            res = {
                "Date": today_str,
                "銘柄名": name,
                "今日の終値": round(float(current_price), 1),
                "明日の予測値": round(float(pred_price), 1),
                "明日の上昇確率": round(float(prob_up * 100), 2)
            }
            res.update(tb_results) # 1W〜1Yの結果を合体
            all_results.append(res)
            print(f"✅ {name} 完了 (明日: {res['明日の上昇確率']}%, 1M: {res.get('1M 利確(>10%)', 0)}%)")
            
        except Exception as e:
            print(f"⚠️ {name} の処理中にエラー: {e}")

    # CSV保存処理
    if all_results:
        df = pd.DataFrame(all_results)
        csv_file = 'prediction_history.csv'
        write_header = True
        write_mode = 'w'
        
        if os.path.exists(csv_file):
            try:
                existing_df = pd.read_csv(csv_file, nrows=1)
                if len(existing_df.columns) > 1:
                    write_header = False
                    write_mode = 'a'
            except Exception:
                pass
                
        df.to_csv(csv_file, mode=write_mode, header=write_header, index=False, encoding='utf-8-sig')
        print(f"✨ CSVに {len(all_results)} 件を保存しました。")

if __name__ == "__main__":
    run_daily_batch()