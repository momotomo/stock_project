import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import os
from datetime import datetime, timedelta

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

def run_daily_batch():
    # 日本時間の今日の日付を取得
    jst_now = datetime.utcnow() + timedelta(hours=9)
    today_str = jst_now.strftime('%Y-%m-%d')
    print(f"[{today_str}] 🚀 日次バッチ処理を開始します...")

    # tickers.txt から監視リストを読み込む
    tickers = {}
    if os.path.exists("tickers.txt"):
        print("📁 tickers.txt を読み込み中...")
        with open("tickers.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if ',' in line:
                    name, ticker = line.split(',', 1)
                    tickers[name.strip()] = ticker.strip()
    
    if not tickers:
        print("⚠️ 監視リストが空のため、デフォルト設定を使用します。")
        tickers = {"トヨタ自動車": "7203.T", "三菱UFJ": "8306.T", "ソニーG": "6758.T"}

    all_results = []
    
    # --- 🌐 マクロ経済データの取得 ---
    print("🌐 マクロ経済データを取得中...")
    macro_tickers = {"USDJPY=X": "USDJPY_Ret", "^GSPC": "SP500_Ret"}
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
        print(f"📊 {name} ({ticker}) の分析を開始...")
        try:
            # 個別株データ取得
            data = yf.download(ticker, period="5y", progress=False)
            if data.empty:
                print(f"❌ {name}: データなし")
                continue
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data.loc[:, ~data.columns.duplicated()].copy()
            data.index = pd.to_datetime(data.index).map(lambda x: x.replace(tzinfo=None).normalize())
            
            # 特徴量エンジニアリング
            data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
            data['Disparity_5'] = (data['Close'] / data['Close'].rolling(5).mean()) - 1
            data['Disparity_25'] = (data['Close'] / data['Close'].rolling(25).mean()) - 1
            data['SMA_Cross'] = (data['Close'].rolling(5).mean() / data['Close'].rolling(25).mean()) - 1
            
            delta = data['Close'].diff()
            up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
            ema_up, ema_down = up.ewm(com=13, adjust=False).mean(), down.ewm(com=13, adjust=False).mean()
            data['RSI_14'] = 100 - (100 / (1 + (ema_up / ema_down)))
            
            std_20, ma_20 = data['Close'].rolling(20).std(), data['Close'].rolling(20).mean()
            data['BB_PctB'] = (data['Close'] - (ma_20 - 2*std_20)) / (4*std_20)
            data['BB_Bandwidth'] = (4*std_20) / data['Close'].rolling(25).mean()
            data['MACD_Norm'] = (data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()) / data['Close']
            
            tr = pd.concat([data['High']-data['Low'], abs(data['High']-data['Close'].shift()), abs(data['Low']-data['Close'].shift())], axis=1).max(axis=1)
            data['ATR'] = tr.rolling(14).mean() / data['Close']
            data['OBV_Ret'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum().pct_change()
            
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            if not macro_returns.empty:
                data = data.join(macro_returns)
            for col in ['USDJPY_Ret', 'SP500_Ret']:
                if col not in data.columns: data[col] = 0.0
            data[['USDJPY_Ret', 'SP500_Ret']] = data[['USDJPY_Ret', 'SP500_Ret']].ffill().fillna(0)
            
            # ターゲット作成
            data['Target_Class'] = (data['Close'].shift(-1) > data['Close']).astype(int)
            data['Target_Price'] = data['Close'].shift(-1)
            
            features = ['Log_Return', 'Disparity_5', 'Disparity_25', 'SMA_Cross', 'RSI_14', 
                        'BB_PctB', 'BB_Bandwidth', 'MACD_Norm', 'ATR', 'OBV_Ret', 'USDJPY_Ret', 'SP500_Ret']
            
            data_features = data.dropna(subset=features)
            if len(data_features) < 100: continue
            
            latest_data = data_features.iloc[[-1]]
            current_price = latest_data['Close'].values[0]
            historical_data = data_features.dropna(subset=['Target_Class', 'Target_Price'])
            
            if len(historical_data) < 100: continue
            
            X_raw = historical_data[features]
            scaler = RobustScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), index=X_raw.index, columns=features)
            latest_X_scaled = pd.DataFrame(scaler.transform(latest_data[features]), index=latest_data.index, columns=features)
            
            # モデル学習・予測（明日）
            clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            clf.fit(X_scaled, historical_data['Target_Class'])
            prob_up = clf.predict_proba(latest_X_scaled)[0][1]
            
            reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            reg.fit(X_scaled, historical_data['Target_Price'])
            pred_price = reg.predict(latest_X_scaled)[0]
            
            # トリプルバリア予測（1ヶ月後）
            pt_sl_1m = 0.10
            target_1m = 'Target_TB_1M'
            data[target_1m] = calc_triple_barrier(data['Close'], 21, pt_sl_1m, pt_sl_1m)
            valid_idx = data[data[target_1m].notna()].index.intersection(X_raw.index)
            
            prob_1m = 0.0
            if len(valid_idx) > 100:
                clf_1m = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                clf_1m.fit(X_scaled.loc[valid_idx], data.loc[valid_idx, target_1m])
                cls = list(clf_1m.classes_)
                prob_1m = clf_1m.predict_proba(latest_X_scaled)[0][cls.index(1)] if 1 in cls else 0.0

            # 結果格納
            res = {
                "Date": today_str,
                "銘柄名": name,
                "今日の終値": round(float(current_price), 1),
                "明日の予測値": round(float(pred_price), 1),
                "明日の上昇確率": round(float(prob_up * 100), 2),
                "1M 利確(>10%)": round(float(prob_1m * 100), 2)
            }
            all_results.append(res)
            print(f"✅ {name} の予測成功")
            
        except Exception as e:
            print(f"⚠️ {name} の処理中にエラー: {e}")

    # CSVへの追記（最強のヘッダー判定）
    if all_results:
        df = pd.DataFrame(all_results)
        csv_file = 'prediction_history.csv'
        
        write_header = True
        write_mode = 'w'
        
        # ファイルが存在する場合、本当に中身が有効かチェックする
        if os.path.exists(csv_file):
            try:
                existing_df = pd.read_csv(csv_file, nrows=1)
                # 列名がちゃんと複数読み込めたら、既存データありとみなして「追記モード」にする
                if len(existing_df.columns) > 1:
                    write_header = False
                    write_mode = 'a'
            except Exception:
                # 空ファイルや、見えない改行文字だけの場合はエラーになるので「新規上書きモード」のまま
                pass
                
        df.to_csv(csv_file, mode=write_mode, header=write_header, index=False, encoding='utf-8-sig')
        
        if write_mode == 'w':
            print(f"✨ 新規ファイルとして '{csv_file}' を作成し、ヘッダー付きで {len(all_results)} 件を記録しました。")
        else:
            print(f"✨ 既存の '{csv_file}' に全 {len(all_results)} 件を追記しました。")
    else:
        print("⚠️ 記録するデータがありませんでした。")

if __name__ == "__main__":
    run_daily_batch()