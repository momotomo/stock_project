# ==============================================================================
# Kaggle専用：AI自動学習スクリプト (V2.1 実運用対応・リーク排除版)
# ==============================================================================
import os
import random
import tempfile


def ensure_notebook_dependencies():
    if not os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
        return
    try:
        from IPython import get_ipython
    except Exception:
        return

    shell = get_ipython()
    if shell is None:
        return

    missing = []
    for module_name in ("yfinance", "optuna", "xgboost", "lightgbm"):
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
    if missing:
        shell.system(f"pip install {' '.join(missing)} -q")


ensure_notebook_dependencies()

import csv
import hashlib
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
import json
from pathlib import Path

warnings.simplefilter('ignore', ResourceWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print(f"🚀 [{datetime.datetime.now()}] AI学習パイプライン(V2.1)を起動します...")

# --- 1. 日経225銘柄リスト ---
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
            data['RSI_14'] = 100 - (100 / (1 + (ema_up / (ema_down + 1e-9))))
            
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
            
            # --- V2.1: TargetをOpen→Close（翌日の日中リターン）へ変更 ---
            data["Next_Open"] = data["Open"].shift(-1)
            data["Next_Close"] = data["Close"].shift(-1)

            # 日中リターン（翌日）
            data["Target_Return"] = (data["Next_Close"] / data["Next_Open"]) - 1.0

            # クラスも「翌日の日中が上か」
            data["Target_Class"] = (data["Next_Close"] > data["Next_Open"]).astype(int)

            # --- V2.1: ATR_Prev（bfill禁止）---
            data["ATR_Prev_Ratio"] = data["ATR"].shift(1)
            
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

# V2.1: dropna に ATR_Prev_Ratio を追加
df_panel = df_panel.dropna(subset=ALL_FEATURES + ['Target_Return', 'Target_Class', 'ATR_Prev_Ratio'])
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
fs_model = lgb.LGBMRanker(n_estimators=100, random_state=SEED, verbose=-1)
fs_model.fit(X_train_scaled, y_train_rank, group=group_train)
importance = pd.Series(fs_model.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
top_k = min(int(len(ALL_FEATURES) * 0.8), len(ALL_FEATURES))
selected_features = importance.head(top_k).index.tolist()
X_train_sel = X_train_scaled[selected_features]

# 🔥 変更: Optunaの週1回化と best_params.json の安全な運用
PARAMS_IN = Path("/kaggle/input/stock-project-params/best_params.json")
PARAMS_OUT = Path("best_params.json")
PARAMS_LOCAL = Path("best_params.json")
TRAINING_RUN_LOG_PATH = Path("training_run_log.csv")
DEFAULT_TUNING_WEEKDAY = 5  # 5=土曜日
DEFAULT_OPTUNA_TRIALS = 15
FORCE_WEEKLY_TUNING = os.getenv("FORCE_WEEKLY_TUNING", "").strip().lower() in {"1", "true", "yes", "on"}
TRAINING_RUN_LOG_HEADER = [
    "run_at",
    "strategy_version",
    "params_version",
    "optuna_executed",
    "n_trials",
    "params_source",
    "train_start",
    "train_end",
    "valid_start",
    "valid_end",
    "oos_start",
    "oos_end",
    "selected_metric",
    "best_params_path",
    "adoption_decision",
]
STRATEGY_VERSION = "V2.1_weekly_optuna"
SELECTED_METRIC = "ndcg@1"

FALLBACK_RANKER_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "max_depth": 4,
    "num_leaves": 31,
    "random_state": SEED,
    "verbose": -1,
}
REQUIRED_RANKER_PARAM_KEYS = {"n_estimators", "learning_rate", "max_depth", "num_leaves"}

def jst_now():
    return datetime.datetime.utcnow() + datetime.timedelta(hours=9)

def get_tuning_weekday():
    try:
        weekday = int(os.getenv("OPTUNA_TUNING_WEEKDAY", str(DEFAULT_TUNING_WEEKDAY)))
    except ValueError:
        weekday = DEFAULT_TUNING_WEEKDAY
    return weekday if 0 <= weekday <= 6 else DEFAULT_TUNING_WEEKDAY

def get_optuna_trials():
    try:
        trials = int(os.getenv("OPTUNA_N_TRIALS", str(DEFAULT_OPTUNA_TRIALS)))
    except ValueError:
        trials = DEFAULT_OPTUNA_TRIALS
    return max(trials, 1)

def is_tuning_day(current_time=None):
    if FORCE_WEEKLY_TUNING:
        return True
    current_time = current_time or jst_now()
    return current_time.weekday() == get_tuning_weekday()

def normalize_ranker_params(params):
    if not isinstance(params, dict):
        return None
    normalized = FALLBACK_RANKER_PARAMS.copy()
    for key, value in params.items():
        if value in (None, ""):
            continue
        normalized[key] = value
    if any(normalized.get(key) in (None, "") for key in REQUIRED_RANKER_PARAM_KEYS):
        return None
    return normalized

def iter_param_sources():
    seen = set()
    for path in (PARAMS_LOCAL, PARAMS_IN):
        path_key = str(path)
        if path_key in seen:
            continue
        seen.add(path_key)
        yield path

def load_best_params():
    for path in iter_param_sources():
        if not path.exists():
            continue
        try:
            params = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"⚠️ パラメータ読み込みエラー: path={path} error={exc}")
            continue
        normalized = normalize_ranker_params(params)
        if normalized is None:
            print(f"⚠️ パラメータ内容が不正または空です: path={path}")
            continue
        return normalized, str(path)
    return None, ""

def save_best_params(params: dict):
    normalized = normalize_ranker_params(params)
    if normalized is None:
        raise ValueError("best_params.json に保存できない不正な params です")
    temp_path = Path(f"{PARAMS_OUT}.tmp")
    temp_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(PARAMS_OUT)
    print(f"💾 best_params.json を更新しました: {PARAMS_OUT}")


def build_date_time_series_splits(index, n_splits=5):
    unique_dates = list(pd.Index(index).unique())
    if len(unique_dates) < 2:
        return []

    if len(unique_dates) == 2:
        train_mask = index.isin(unique_dates[:1])
        val_mask = index.isin(unique_dates[1:])
        return [(np.flatnonzero(train_mask), np.flatnonzero(val_mask))]

    effective_splits = min(n_splits, len(unique_dates) - 1)
    splitter = TimeSeriesSplit(n_splits=effective_splits)
    splits = []
    unique_dates_array = np.array(unique_dates)
    for train_date_idx, val_date_idx in splitter.split(unique_dates_array):
        train_dates = unique_dates_array[train_date_idx]
        val_dates = unique_dates_array[val_date_idx]
        train_mask = index.isin(train_dates)
        val_mask = index.isin(val_dates)
        splits.append((np.flatnonzero(train_mask), np.flatnonzero(val_mask)))
    return splits


def atomic_joblib_dump(obj, path):
    target_path = Path(path)
    temp_handle = tempfile.NamedTemporaryFile(
        delete=False,
        dir=str(target_path.parent),
        prefix=f".{target_path.name}.",
        suffix=".tmp",
    )
    temp_handle.close()
    temp_path = Path(temp_handle.name)
    try:
        joblib.dump(obj, temp_path)
        temp_path.replace(target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

def run_weekly_optuna():
    trial_count = get_optuna_trials()
    print(
        f"⚙️ 週次 Optuna を実行します: weekday={get_tuning_weekday()} "
        f"trials={trial_count}"
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 10, 60),
            "random_state": SEED,
            "verbose": -1,
        }
        dates = sorted(train_df.index.unique())
        if len(dates) < 2:
            return 0.0
        split_idx = int(len(dates) * 0.8)
        mask_train = train_df.index < dates[split_idx]
        mask_val = train_df.index >= dates[split_idx]

        X_t, y_t = X_train_sel[mask_train], y_train_rank[mask_train]
        g_t = train_df[mask_train].groupby(level=0).size().values
        X_v, y_v = X_train_sel[mask_val], y_train_rank[mask_val]
        g_v = train_df[mask_val].groupby(level=0).size().values

        if len(g_t) == 0 or len(g_v) == 0:
            return 0.0
        model = lgb.LGBMRanker(**params)
        model.fit(
            X_t,
            y_t,
            group=g_t,
            eval_set=[(X_v, y_v)],
            eval_group=[g_v],
            callbacks=[lgb.early_stopping(10, verbose=False)],
        )
        return model.best_score_["valid_0"]["ndcg@1"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trial_count)
    best_params = normalize_ranker_params(study.best_params)
    if best_params is None:
        raise ValueError("Optuna が不正または空の params を返しました")
    return best_params

def classify_params_source(source_detail):
    if source_detail.startswith("weekly_optuna"):
        return "weekly_optuna"
    if source_detail.startswith("cached"):
        return "cached"
    if source_detail.startswith("fallback"):
        return "fallback"
    return source_detail or "unknown"

def resolve_best_params_path(source_detail):
    if source_detail.startswith("cached:") or source_detail.startswith("cached_after_optuna_failure:"):
        return source_detail.split(":", 1)[1]
    if classify_params_source(source_detail) == "weekly_optuna":
        return str(PARAMS_OUT)
    return ""

def build_params_version(params):
    payload = json.dumps(params, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

def build_adoption_decision(source_detail, optuna_executed):
    if source_detail.startswith("cached_after_optuna_failure"):
        return "adopt_cached_after_optuna_failure"
    if source_detail.startswith("fallback_after_optuna_failure"):
        return "adopt_fallback_after_optuna_failure"
    canonical = classify_params_source(source_detail)
    if canonical == "weekly_optuna":
        return "adopt_weekly_optuna" if optuna_executed else "weekly_optuna_not_executed"
    if canonical == "cached":
        return "adopt_cached"
    if canonical == "fallback":
        return "adopt_fallback"
    return "adopt_unknown"

def format_date_value(value):
    if value is None:
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)

def build_training_window_summary(train_dates, oos_date):
    train_dates = list(train_dates)
    if not train_dates:
        return {
            "train_start": "",
            "train_end": "",
            "valid_start": "",
            "valid_end": "",
            "oos_start": format_date_value(oos_date),
            "oos_end": format_date_value(oos_date),
        }

    split_idx = int(len(train_dates) * 0.8)
    split_idx = max(1, min(split_idx, len(train_dates) - 1)) if len(train_dates) > 1 else 0
    valid_dates = train_dates[split_idx:] if split_idx < len(train_dates) else []

    return {
        "train_start": format_date_value(train_dates[0]),
        "train_end": format_date_value(train_dates[-1]),
        "valid_start": format_date_value(valid_dates[0]) if valid_dates else "",
        "valid_end": format_date_value(valid_dates[-1]) if valid_dates else "",
        "oos_start": format_date_value(oos_date),
        "oos_end": format_date_value(oos_date),
    }

def append_training_run_log(row):
    file_exists = TRAINING_RUN_LOG_PATH.exists() and TRAINING_RUN_LOG_PATH.stat().st_size > 0
    with TRAINING_RUN_LOG_PATH.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAINING_RUN_LOG_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in TRAINING_RUN_LOG_HEADER})

def resolve_ranker_params():
    cached_params, cached_source = load_best_params()

    if is_tuning_day():
        if FORCE_WEEKLY_TUNING:
            print("⚙️ FORCE_WEEKLY_TUNING=1 のため、曜日に関係なく Optuna を実行します。")
        else:
            print("⚙️ 本日はチューニング曜日です。Optuna を実行します。")
        try:
            tuned_params = run_weekly_optuna()
            save_best_params(tuned_params)
            return tuned_params, "weekly_optuna", True, get_optuna_trials()
        except Exception as exc:
            print(f"⚠️ Optuna 実行に失敗しました。既存 params を優先し、best_params.json は更新しません: {exc}")
            if cached_params is not None:
                return cached_params, f"cached_after_optuna_failure:{cached_source}", True, get_optuna_trials()
            return FALLBACK_RANKER_PARAMS.copy(), "fallback_after_optuna_failure", True, get_optuna_trials()

    if cached_params is not None:
        print(f"⚙️ 平日実行: Optuna をスキップし、既存 params を利用します: {cached_source}")
        return cached_params, f"cached:{cached_source}", False, 0

    print("⚙️ 平日実行: best_params.json が無いため fallback params を利用します。")
    return FALLBACK_RANKER_PARAMS.copy(), "fallback", False, 0

best_params, best_params_source_detail, optuna_executed, optuna_trials = resolve_ranker_params()
best_params_source = classify_params_source(best_params_source_detail)
best_params_path = resolve_best_params_path(best_params_source_detail)
train_dates = sorted(train_df.index.unique())
training_window = build_training_window_summary(train_dates, latest_date)
training_run_row = {
    "run_at": jst_now().strftime("%Y-%m-%d %H:%M:%S"),
    "strategy_version": STRATEGY_VERSION,
    "params_version": build_params_version(best_params),
    "optuna_executed": str(bool(optuna_executed)),
    "n_trials": optuna_trials,
    "params_source": best_params_source,
    "train_start": training_window["train_start"],
    "train_end": training_window["train_end"],
    "valid_start": training_window["valid_start"],
    "valid_end": training_window["valid_end"],
    "oos_start": training_window["oos_start"],
    "oos_end": training_window["oos_end"],
    "selected_metric": SELECTED_METRIC,
    "best_params_path": best_params_path,
    "adoption_decision": build_adoption_decision(best_params_source_detail, optuna_executed),
}
append_training_run_log(training_run_row)
print(
    f"🧭 Ranker params source={best_params_source} optuna_executed={optuna_executed} "
    f"best_params_path={best_params_path or 'N/A'}"
)

# --- 3. 最終モデルの学習 (Ranker & ベースXGBoost & メタLightGBM & Regressor) ---
print("🧠 最終モデル(Ranker)を学習中...")
ranker_model = lgb.LGBMRanker(**best_params)
ranker_model.fit(X_train_sel, y_train_rank, group=group_train)

print("🔍 メタラベリング用 OOF(クロスバリデーション)予測を生成中...")
oof_probs = np.zeros(len(X_train_sel))
time_splits = build_date_time_series_splits(X_train_sel.index, n_splits=5)

for train_index, val_index in time_splits:
    X_tr, y_tr = X_train_sel.iloc[train_index], y_train_class.iloc[train_index]
    X_va = X_train_sel.iloc[val_index]
    xgb_cv = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        random_state=SEED,
        eval_metric='logloss',
    )
    xgb_cv.fit(X_tr, y_tr)
    oof_probs[val_index] = xgb_cv.predict_proba(X_va)[:, 1]

print("🧠 メタモデル(確信度AI)を学習中...")
X_meta = X_train_sel.copy()
X_meta['Base_Prob'] = oof_probs
meta_model = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=SEED, verbose=-1)
meta_model.fit(X_meta, y_train_class)

print("🧠 本番用ベースモデル(XGBoost)を学習中...")
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=SEED, eval_metric='logloss')
xgb_model.fit(X_train_sel, y_train_class)

# --- V2.1: 回帰モデル（Pred_Return用）の学習を追加 ---
print("🧠 本番用リターン回帰モデル(LGBMRegressor)を学習中...")
y_train_reg = train_df["Target_Return"].astype(float)
regressor_model = lgb.LGBMRegressor(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=4,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    objective="regression", # huberが弾かれる環境対策としてregressionを採用
    verbose=-1,
)
regressor_model.fit(X_train_sel, y_train_reg)

# --- 4. モデルの保存 (KaggleのOutputとして保存) ---
print("💾 モデルをファイルに保存中(Kaggle Output)...")
atomic_joblib_dump(ranker_model, 'ranker_model.pkl')
atomic_joblib_dump(xgb_model, 'classifier_model.pkl')
atomic_joblib_dump(meta_model, 'meta_model.pkl')
atomic_joblib_dump(regressor_model, 'regressor_model.pkl') # V2.1で追加
atomic_joblib_dump(scaler, 'scaler.pkl')
atomic_joblib_dump(selected_features, 'selected_features.pkl')

print(f"🎉 [{datetime.datetime.now()}] 全ての処理が完了しました！ファイルはKaggleのOutputとして保存されました。")
