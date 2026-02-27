import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
import os
from datetime import datetime, timedelta
import warnings

# 無害な警告を非表示にする
warnings.simplefilter('ignore', ResourceWarning)

# --- おすすめ候補の大型優良株30銘柄（TOPIX Core30相当） ---
RECOMMEND_POOL = {
    "トヨタ自動車": "7203.T", "ソニーG": "6758.T", "三菱UFJ": "8306.T",
    "キーエンス": "6861.T", "NTT": "9432.T", "ファーストリテイリング": "9983.T",
    "東京エレクトロン": "8035.T", "信越化学": "4063.T", "三井住友FG": "8316.T",
    "日立製作所": "6501.T", "伊藤忠商事": "8001.T", "KDDI": "9433.T",
    "ホンダ": "7267.T", "三菱商事": "8058.T", "ソフトバンクG": "9984.T",
    "三井物産": "8031.T", "ダイキン工業": "6367.T", "武田薬品": "4502.T",
    "リクルートHD": "6098.T", "任天堂": "7974.T", "みずほFG": "8411.T",
    "村田製作所": "6981.T", "デンソー": "6902.T", "ファナック": "6954.T",
    "アステラス製薬": "4503.T", "セブン＆アイ": "3382.T", "オリックス": "8591.T",
    "第一三共": "4568.T", "コマツ": "6301.T", "丸紅": "8002.T"
}

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

# AIの数値に基づく「おすすめ理由」の自動生成ロジック
def generate_reason(row_data, probs):
    reason = []
    if probs['明日'] > 60:
        reason.append("明日の上昇確率が高く、短期的な勢いが非常に強い状態です。")
    elif probs['1W'] > 60:
        reason.append("数日〜1週間程度のスイングトレードで利益が狙いやすいチャート形状です。")
    else:
        reason.append("短期的な過熱感はなく、押し目買いの好機となる可能性があります。")
    
    if probs['1M'] > 60 or probs['3M'] > 60:
        reason.append("1ヶ月〜3ヶ月の中期的な上昇トレンドがAIにより強く支持されています。")
    
    rsi = row_data['RSI_14'].values[0]
    if rsi < 30:
        reason.append(f"現在RSIが{rsi:.1f}と売られ過ぎ水準にあり、反発が見込めます。")
    elif rsi > 70:
        reason.append(f"現在RSIが{rsi:.1f}と強く買われており、勢いに乗る順張りが有効です。")
    else:
        reason.append(f"RSIは{rsi:.1f}で過熱感のないニュートラルな水準であり、安定した値動きが想定されます。")
        
    return " ".join(reason)

def run_daily_batch():
    # 日本時間の今日の日付を取得
    jst_now = datetime.utcnow() + timedelta(hours=9)
    today_str = jst_now.strftime('%Y-%m-%d')
    print(f"[{today_str}] 🚀 日次バッチ処理（全スパン ＆ おすすめ発掘機能付き）を開始します...")

    # tickers.txt から監視リストを読み込む
    user_tickers = {}
    if os.path.exists("tickers.txt"):
        with open("tickers.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if ',' in line:
                    name, ticker = line.split(',', 1)
                    user_tickers[name.strip()] = ticker.strip()
    
    if not user_tickers:
        user_tickers = {"トヨタ自動車": "7203.T", "三菱UFJ": "8306.T", "ソニーG": "6758.T"}

    # 対象の全銘柄（おすすめプール ＋ ユーザー指定）
    target_tickers = {}
    target_tickers.update(RECOMMEND_POOL)
    target_tickers.update(user_tickers)

    # --- マクロ経済データの取得 ---
    print("🌐 マクロ経済データ(為替, SP500)を取得中...")
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
        macro_returns = pd.DataFrame()

    user_results = []
    recommend_results = []
    
    # 全スパン設定
    tb_horizons = {'1W': 5, '2W': 10, '1M': 21, '3M': 63, '6M': 126, '1Y': 252}
    pt_sl = {'1W': 0.03, '2W': 0.05, '1M': 0.10, '3M': 0.20, '6M': 0.30, '1Y': 0.50}

    for name, ticker in target_tickers.items():
        print(f"📊 {name} ({ticker}) を分析中...")
        try:
            data = yf.download(ticker, period="5y", progress=False)
            if data.empty: continue
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data.loc[:, ~data.columns.duplicated()].copy()
            data.index = pd.to_datetime(data.index).map(lambda x: x.replace(tzinfo=None).normalize())
            
            # 特徴量エンジニアリング (元の12特徴量を維持)
            data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
            data['Disparity_5'] = (data['Close'] / data['Close'].rolling(5).mean()) - 1
            data['Disparity_25'] = (data['Close'] / data['Close'].rolling(25).mean()) - 1
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
            
            tr = pd.concat([data['High']-data['Low'], abs(data['High']-data['Close'].shift()), abs(data['Low']-data['Close'].shift())], axis=1).max(axis=1)
            data['ATR'] = tr.rolling(14).mean() / data['Close']
            data['OBV_Ret'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum().pct_change()
            
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            if not macro_returns.empty: data = data.join(macro_returns)
            for col in ['USDJPY_Ret', 'SP500_Ret']:
                if col not in data.columns: data[col] = 0.0
            data[['USDJPY_Ret', 'SP500_Ret']] = data[['USDJPY_Ret', 'SP500_Ret']].ffill().fillna(0)
            
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
            
            # --- 学習と予測 ---
            clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            y_train_tom = historical_data['Target_Class']
            if len(np.unique(y_train_tom)) > 1:
                clf.fit(X_scaled, y_train_tom)
                prob_up = clf.predict_proba(latest_X_scaled)[0][1]
            else:
                prob_up = 0.0
                
            reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            reg.fit(X_scaled, historical_data['Target_Price'])
            pred_price = reg.predict(latest_X_scaled)[0]
            
            # 全スパンの確率計算
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
            
            # スコアとおすすめ理由の生成
            short_term_score = np.mean([prob_up, tb_results['1W'], tb_results['2W']])
            mid_long_term_score = np.mean([tb_results['1M'], tb_results['3M'], tb_results['6M'], tb_results['1Y']])
            reason = generate_reason(latest_data, {'明日': prob_up*100, '1W': tb_results['1W']*100, '1M': tb_results['1M']*100, '3M': tb_results['3M']*100})
            
            res = {
                "Date": today_str,
                "銘柄名": name,
                "今日の終値": round(float(current_price), 1),
                "明日の予測値": round(float(pred_price), 1),
                "明日の上昇確率": round(float(prob_up * 100), 2),
                "短期スコア": round(float(short_term_score * 100), 2),
                "中長期スコア": round(float(mid_long_term_score * 100), 2),
                "おすすめ理由": reason
            }
            for k in tb_horizons.keys():
                res[f"{k} 利確(>{int(pt_sl[k]*100)}%)"] = round(float(tb_results[k] * 100), 2)
            
            if name in user_tickers: user_results.append(res)
            if name in RECOMMEND_POOL: recommend_results.append(res)
            
        except Exception as e:
            print(f"⚠️ {name} の処理中にエラー: {e}")

    # 1. ユーザー監視リストの履歴保存 (同日上書き機能付き)
    if user_results:
        save_keys = ["Date", "銘柄名", "今日の終値", "明日の予測値", "明日の上昇確率"] + [f"{k} 利確(>{int(pt_sl[k]*100)}%)" for k in tb_horizons.keys()]
        df_new = pd.DataFrame([{k: v for k, v in r.items() if k in save_keys} for r in user_results])
        csv_file = 'prediction_history.csv'
        
        if os.path.exists(csv_file):
            try:
                df_existing = pd.read_csv(csv_file)
                # 同じ日付・銘柄のデータを削除して新しいデータで上書きする
                cond = (df_existing['Date'] == today_str) & (df_existing['銘柄名'].isin(df_new['銘柄名']))
                df_existing = df_existing[~cond]
                df_final = pd.concat([df_existing, df_new], ignore_index=True)
                df_final.to_csv(csv_file, index=False, encoding='utf-8-sig')
                print(f"✨ 監視履歴を上書き更新しました ({len(user_results)} 件)")
            except Exception: 
                df_new.to_csv(csv_file, index=False, encoding='utf-8-sig')
        else:
            df_new.to_csv(csv_file, index=False, encoding='utf-8-sig')

    # 2. おすすめ銘柄の保存
    if recommend_results:
        pd.DataFrame(recommend_results).to_csv('recommendations.csv', index=False, encoding='utf-8-sig')
        print(f"✨ AIおすすめ銘柄を更新しました！")

if __name__ == "__main__":
    run_daily_batch()