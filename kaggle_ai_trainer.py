# ==============================================================================
# Kaggle専用：AI自動学習スクリプト (API直結・GitHub不要版)
# ==============================================================================
!pip install yfinance optuna xgboost lightgbm -q

import os
import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import optuna
import joblib
import warnings
import datetime
import time

warnings.simplefilter('ignore', ResourceWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

print(f"🚀 [{datetime.datetime.now()}] AI学習パイプライン(API直結版)を起動します...")

# --- 1. 日経225銘柄リスト（安定版：直接定義） ---
def get_nikkei225_tickers():
    print("🌐 日経225主要銘柄リストを読み込みます...")
    codes = [
        4151, 4502, 4503, 4519, 4523, 4568, 4578, 6098, 7741, 1332, 1605, 1925, 1928, 2502, 
        2801, 2802, 2914, 3382, 3401, 3402, 3405, 3407, 3861, 3863, 4004, 4005, 4021, 4042, 
        4043, 4063, 4183, 4188, 4208, 4452, 4631, 4901, 4911, 5019, 5020, 5101, 5108, 5201, 
        5214, 5301, 5332, 5333, 5401, 5406, 5411, 5631, 5703, 5711, 5713, 5714, 5801, 5802, 
        5803, 5901, 6103, 6118, 6301, 6302, 6305, 6326, 6367, 6471, 6472, 6473, 6501, 6503, 
        6504, 6506, 6645, 6701, 6702, 6724, 6752, 6753, 6758, 6762, 6770, 6841, 6857, 6861, 
        6902, 6954, 6971, 6981, 6988, 7003, 7011, 7012, 7013, 7186, 7201, 7202, 7203, 7205, 
        7211, 7259, 7261, 7267, 7269, 7270, 7272, 7731, 7733, 7735, 7751, 7752, 7911, 7912, 
        7951, 8015, 8031, 8035, 8053, 8058, 9766, 2432, 3659, 4324, 4689, 4704, 4751, 7974, 
        8001, 8002, 8028, 8766, 9432, 9433, 9434, 9613, 9735, 9843, 9983, 9984, 8252, 8253, 
        8303, 8304, 8306, 8308, 8309, 8316, 8331, 8354, 8355, 8411, 8550, 8586, 8591, 8593, 
        8601, 8604, 8628, 8630, 8725, 8750, 8795, 8801, 8802, 8804, 8830, 8892, 8953, 9001, 
        9005, 9007, 9008, 9009, 9020, 9021, 9022, 9062, 9064, 9101, 9104, 9107, 9147, 9201, 
        9202, 9301, 9501, 9502, 9503, 9504, 9506, 9508, 9531, 9532, 9602, 9736
    ]
    tickers = {f"Code_{code}": f"{code}.T" for code in codes}
    print(f"✅ {len(tickers)} 銘柄のリストを読み込みました。")
    return tickers

TICKERS = get_nikkei225_tickers()

# --- 2. データ取得と特徴量生成 ---
def get_macro_data():
    macro_tickers = {"USDJPY=X": "USDJPY_Ret", "^GSPC": "SP500_Ret", "1306.T": "TOPIX_Ret"}
    macro_df = yf.download(list(macro_tickers.keys()), period="5y", progress=False)
    macro_data = macro_df.xs('Close', level=0, axis=1) if isinstance(macro_df.columns, pd.MultiIndex) else macro_df['Close']
    macro_returns = np.log(macro_data / macro_data.shift(1)).rename(columns=macro_tickers)
    macro_returns.index = pd.to_datetime(macro_returns.index).map(lambda x: x.replace(tzinfo=None).normalize())
    return macro_returns

def calc_fractional_diff(series, d=0.5, window=20):
    weights = [1.0]
    for k in range(1, window):
        weights.append(-weights[-1] * (d - k + 1) / k)
    weights = np.array(weights)[::-1]
    return series.rolling(window).apply(lambda x: np.dot(weights, x), raw=True)

def get_stock_features(ticker, macro_returns):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            data = yf.download(ticker, period="5y", progress=False)
            if data.empty: return None
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
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
            ema_up = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
            ema_down = (-1 * delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
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

            for col in ['Log_Return_Norm', 'RSI_14', 'MACD_Norm']:
                data[f'{col}_Lag1'] = data[col].shift(1)
                data[f'{col}_Lag2'] = data[col].shift(2)
                
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            if not macro_returns.empty: data = data.join(macro_returns)
            for col in ['USDJPY_Ret', 'SP500_Ret', 'TOPIX_Ret']:
                if col not in data.columns: data[col] = 0.0
            data[['USDJPY_Ret', 'SP500_Ret', 'TOPIX_Ret']] = data[['USDJPY_Ret', 'SP500_Ret', 'TOPIX_Ret']].ffill().fillna(0)
            data['Excess_Return'] = data['Log_Return'] - data['TOPIX_Ret']
            data['Excess_Return_20d'] = data['Excess_Return'].rolling(20).sum()
            
            data['Target_Class'] = (data['Close'].shift(-1) > data['Close']).astype(int)
            data['Target_Return'] = data['Close'].shift(-1) / data['Close'] - 1
            return data
        except Exception:
            if attempt < max_retries - 1: time.sleep(2)
            else: return None

macro_returns = get_macro_data()
stock_data_dict = {}
count = 0
total = len(TICKERS)

print(f"📊 {total}銘柄のデータ取得を開始します。少し時間がかかります...")
for name, ticker in TICKERS.items():
    data = get_stock_features(ticker, macro_returns)
    if data is not None: stock_data_dict[name] = data
    count += 1
    if count % 20 == 0:
        print(f"  ... {count}/{total} 銘柄処理完了")
        time.sleep(1)

if len(stock_data_dict) < 50:
    print("❌ データが少なすぎます。")
    import sys; sys.exit(1)

rsi_df = pd.DataFrame({name: df['RSI_14'] for name, df in stock_data_dict.items()})
rsi_mean, rsi_std = rsi_df.mean(axis=1), rsi_df.std(axis=1)

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

df_panel = df_panel.dropna(subset=ALL_FEATURES + ['Target_Return', 'Target_Class'])
df_panel['Target_Rank'] = df_panel.groupby(level=0)['Target_Return'].transform(
    lambda x: ((x.rank(method='first') - 1) / max(1, len(x) - 1) * min(4, len(x) - 1)).astype(int)
)

latest_date = df_panel.index.max()
train_df = df_panel[df_panel.index < latest_date]
X_train_raw = train_df[ALL_FEATURES]
y_train_rank = train_df['Target_Rank']
y_train_class = train_df['Target_Class']
group_train = train_df.groupby(level=0).size().values

scaler = RobustScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), index=X_train_raw.index, columns=ALL_FEATURES)

print("🔍 特徴量選択を実行中...")
fs_model = lgb.LGBMRanker(n_estimators=100, random_state=42, verbose=-1)
fs_model.fit(X_train_scaled, y_train_rank, group=group_train)
importance = pd.Series(fs_model.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
top_k = min(int(len(ALL_FEATURES) * 0.8), len(ALL_FEATURES))
selected_features = importance.head(top_k).index.tolist()
X_train_sel = X_train_scaled[selected_features]

print("⚙️ Optunaによるパラメータ最適化を実行中 (15 Trials)...")
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 60),
        'random_state': 42,
        'verbose': -1
    }
    dates = train_df.index.unique()
    split_idx = int(len(dates) * 0.8)
    mask_train = train_df.index < dates[split_idx]
    mask_val = train_df.index >= dates[split_idx]
    
    X_t, y_t = X_train_sel[mask_train], y_train_rank[mask_train]
    g_t = train_df[mask_train].groupby(level=0).size().values
    X_v, y_v = X_train_sel[mask_val], y_train_rank[mask_val]
    g_v = train_df[mask_val].groupby(level=0).size().values
    
    if len(g_t) == 0 or len(g_v) == 0: return 0.0
    model = lgb.LGBMRanker(**params)
    model.fit(X_t, y_t, group=g_t, eval_set=[(X_v, y_v)], eval_group=[g_v], callbacks=[lgb.early_stopping(10, verbose=False)])
    return model.best_score_['valid_0']['ndcg@1']

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15)
best_params = study.best_params
best_params['random_state'] = 42
best_params['verbose'] = -1

# --- 3. 最終モデルの学習 (Ranker & ベースXGBoost & メタLightGBM) ---
print("🧠 最終モデル(Ranker)を学習中...")
ranker_model = lgb.LGBMRanker(**best_params)
ranker_model.fit(X_train_sel, y_train_rank, group=group_train)

print("🔍 メタラベリング用 OOF(クロスバリデーション)予測を生成中...")
tscv = TimeSeriesSplit(n_splits=5)
oof_probs = np.zeros(len(X_train_sel))

for train_index, val_index in tscv.split(X_train_sel):
    X_tr, y_tr = X_train_sel.iloc[train_index], y_train_class.iloc[train_index]
    X_va = X_train_sel.iloc[val_index]
    xgb_cv = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, eval_metric='logloss')
    xgb_cv.fit(X_tr, y_tr)
    oof_probs[val_index] = xgb_cv.predict_proba(X_va)[:, 1]

print("🧠 メタモデル(確信度AI)を学習中...")
X_meta = X_train_sel.copy()
X_meta['Base_Prob'] = oof_probs
meta_model = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1)
meta_model.fit(X_meta, y_train_class)

print("🧠 本番用ベースモデル(XGBoost)を学習中...")
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train_sel, y_train_class)

# --- 4. モデルの保存 (KaggleのOutputとして保存) ---
print("💾 モデルをファイルに保存中(Kaggle Output)...")
# APIでダウンロードしやすいようにカレントディレクトリに保存
joblib.dump(ranker_model, 'ranker_model.pkl')
joblib.dump(xgb_model, 'classifier_model.pkl')
joblib.dump(meta_model, 'meta_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selected_features, 'selected_features.pkl')

print(f"🎉 [{datetime.datetime.now()}] 全ての処理が完了しました！ファイルはKaggleのOutputとして保存されました。")