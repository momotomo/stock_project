# ==============================================================================
# Kaggle専用：AI自動学習＆GitHub転送スクリプト (日経225 + メタラベリング版)
# ==============================================================================
!pip install yfinance optuna PyGithub xgboost lightgbm beautifulsoup4 lxml -q

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
from github import Github
from kaggle_secrets import UserSecretsClient
import warnings
import datetime
import requests
from bs4 import BeautifulSoup
import time

warnings.simplefilter('ignore', ResourceWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

print(f"🚀 [{datetime.datetime.now()}] AI学習パイプライン(日経225・メタラベリング版)を起動します...")

try:
    user_secrets = UserSecretsClient()
    GITHUB_TOKEN = user_secrets.get_secret("GITHUB_TOKEN")
    GITHUB_REPO = user_secrets.get_secret("GITHUB_REPO")
    print("✅ GitHubトークンの読み込みに成功しました。")
except Exception as e:
    print(f"❌ Kaggle Secretsの読み込みエラー: {e}")
    raise e

def get_nikkei225_tickers():
    print("🌐 Wikipediaから日経225の最新構成銘柄を取得中...")
    url = "https://ja.wikipedia.org/wiki/%E6%97%A5%E7%B5%8C%E5%B9%B3%E5%9D%87%E6%A0%AA%E4%BE%A1#%E6%A7%8B%E6%88%90%E9%8A%98%E6%9F%84"
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'lxml')
        tables = soup.find_all('table', class_='wikitable')
        target_table = None
        for table in tables:
            if 'コード' in table.text and '銘柄名' in table.text:
                target_table = table
                break
        if not target_table: raise ValueError("テーブルが見つかりません。")

        tickers = {}
        for row in target_table.find_all('tr')[1:]:
            cols = row.find_all(['td', 'th'])
            if len(cols) >= 2:
                code = cols[0].text.strip()
                name = cols[1].text.strip()
                if code.isdigit(): tickers[name] = f"{code}.T"
        print(f"✅ 日経225銘柄の取得に成功しました。合計: {len(tickers)} 銘柄")
        return tickers
    except Exception as e:
        print(f"❌ 取得失敗。デフォルトを使用します: {e}")
        return {"トヨタ自動車": "7203.T", "三菱UFJ": "8306.T", "ソフトバンクG": "9984.T"}

TICKERS = get_nikkei225_tickers()

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

# --- 6. 最終モデルの学習 (Ranker & ベースXGBoost & メタLightGBM) ---
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

print("💾 モデルをファイルに保存中...")
os.makedirs("models", exist_ok=True)
joblib.dump(ranker_model, 'models/ranker_model.pkl')
joblib.dump(xgb_model, 'models/classifier_model.pkl')
joblib.dump(meta_model, 'models/meta_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(selected_features, 'models/selected_features.pkl')

print("🌐 GitHubリポジトリへモデルを転送中...")
g = Github(GITHUB_TOKEN)
repo = g.get_repo(GITHUB_REPO)

def upload_to_github(file_path, git_path):
    with open(file_path, 'rb') as file: content = file.read()
    try:
        contents = repo.get_contents(git_path)
        repo.update_file(contents.path, f"Auto-update {git_path} from Kaggle", content, contents.sha)
    except Exception:
        repo.create_file(git_path, f"Create {git_path} from Kaggle", content)

upload_to_github('models/ranker_model.pkl', 'models/ranker_model.pkl')
upload_to_github('models/classifier_model.pkl', 'models/classifier_model.pkl')
upload_to_github('models/meta_model.pkl', 'models/meta_model.pkl')
upload_to_github('models/scaler.pkl', 'models/scaler.pkl')
upload_to_github('models/selected_features.pkl', 'models/selected_features.pkl')

print(f"🎉 [{datetime.datetime.now()}] 全ての処理が完了しました！")