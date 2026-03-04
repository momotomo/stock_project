import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
import os
import plotly.express as px
from github import Github
import warnings
import optuna

warnings.simplefilter('ignore', ResourceWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

st.set_page_config(layout="wide", page_title="AI株価スクリーニング")

def calculate_psi(expected, actual, bins=10):
    bins_edges = np.linspace(0, 1, bins + 1)
    expected_percents = np.histogram(expected, bins=bins_edges)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=bins_edges)[0] / len(actual)
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    return np.sum(psi_values)

def get_tickers():
    tickers = {}
    if os.path.exists("tickers.txt"):
        with open("tickers.txt", "r", encoding="utf-8") as f:
            for line in f:
                if ',' in line:
                    name, ticker = line.split(',', 1)
                    tickers[name.strip()] = ticker.strip()
    return tickers

def get_secret(key):
    if key in st.secrets:
        return st.secrets[key]
    secrets_path = os.path.join(".streamlit", "secrets.toml")
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        if k.strip().upper() == key.upper():
                            return v.strip().strip('"').strip("'")
        except Exception:
            pass
    return None

@st.cache_data(show_spinner=False)
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
    except Exception:
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
    
    data['Target_Class'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data['Target_Return'] = data['Close'].shift(-1) / data['Close'] - 1
    return data

@st.cache_data(show_spinner=False)
def fetch_and_prepare_all_data(tickers_dict):
    macro_returns = get_macro_data()
    stock_data_dict = {}
    for name, ticker in tickers_dict.items():
        data = get_stock_features(ticker, macro_returns)
        if data is not None:
            stock_data_dict[name] = data
            
    if stock_data_dict:
        rsi_df = pd.DataFrame({name: df['RSI_14'] for name, df in stock_data_dict.items()})
        rsi_mean = rsi_df.mean(axis=1)
        rsi_std = rsi_df.std(axis=1)
        for name in stock_data_dict.keys():
            stock_data_dict[name]['RSI_Z_Score'] = (stock_data_dict[name]['RSI_14'] - rsi_mean) / (rsi_std + 1e-9)
            
    return stock_data_dict

EXTENDED_FEATURES = [
    'Log_Return_Norm', 'Frac_Diff_0.5', 'Disparity_5_Norm', 'Disparity_25_Norm', 
    'SMA_Cross', 'RSI_14', 'BB_PctB', 'BB_Bandwidth', 'MACD_Norm', 
    'ATR', 'OBV_Ret', 'USDJPY_Ret', 'SP500_Ret', 'TOPIX_Ret',
    'EMA_Diff_Ratio', 'MACD_Div', 'Log_Return_Norm_Lag1', 'Log_Return_Norm_Lag2',
    'RSI_14_Lag1', 'RSI_14_Lag2', 'MACD_Norm_Lag1', 'MACD_Norm_Lag2', 'RSI_Z_Score',
    'Excess_Return', 'Excess_Return_20d'
]

st.title("🚀 AI株価スクリーニングダッシュボード (Optuna自己学習・Ranker版)")

tab1, tab2, tab3 = st.tabs(["🔍 今日の分析 ＆ おすすめ", "📈 予測推移", "📊 バックテスト (検証)"])

st.sidebar.header("⚙️ 分析銘柄の設定")
TICKERS_FILE = "tickers.txt"

def load_tickers_text():
    if os.path.exists(TICKERS_FILE):
        with open(TICKERS_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return "トヨタ自動車, 7203.T\n三菱UFJ, 8306.T"

ticker_input = st.sidebar.text_area(
    "監視リスト", value=load_tickers_text(), height=300,
    help="1行に1銘柄ずつ「銘柄名, ティッカー」の形式で入力"
)

if st.sidebar.button("💾 監視リストを保存・同期", use_container_width=True):
    with open(TICKERS_FILE, "w", encoding="utf-8") as f:
        f.write(ticker_input)
    token = get_secret("GITHUB_TOKEN")
    repo_name = get_secret("GITHUB_REPO")
    
    if token and repo_name:
        try:
            g = Github(token)
            repo = g.get_repo(repo_name)
            try:
                contents = repo.get_contents(TICKERS_FILE)
                repo.update_file(contents.path, "Update tickers from Web UI", ticker_input, contents.sha)
            except Exception:
                repo.create_file(TICKERS_FILE, "Create tickers.txt", ticker_input)
            st.sidebar.success("✅ GitHubに同期しました！")
        except Exception as e:
            st.sidebar.error(f"GitHub同期エラー: {e}")
    else:
        st.sidebar.warning("✅ ローカルに保存しました（クラウド同期未設定）")

with tab1:
    st.subheader("🌟 AIが選ぶ本日のおすすめ銘柄 (自動更新)")
    st.write("設定された銘柄群をLGBMRankerが横断的に評価し、相対的な「強さ」で順位付けしています。")
    
    rec_file = 'recommendations.csv'
    if os.path.exists(rec_file) and os.path.getsize(rec_file) > 0:
        try:
            df_rec = pd.read_csv(rec_file)
            
            # 🔥 追加: データの先頭に「順位」列を追加
            if '順位' not in df_rec.columns:
                df_rec.insert(0, '順位', range(1, len(df_rec) + 1))
            
            st.markdown("### ⚡ 本日のAI推奨トップ3")
            if "短期スコア" in df_rec.columns:
                # 🔥 修正: 余計な再ソートをやめ、AIが書き出した並び順（Rankerの順位）をそのまま採用する
                short_top = df_rec.head(3)
                
                # 🔥 修正: 見切れを防ぎつつ、プロのダッシュボードのようなカード型UIに変更
                cols_s = st.columns(3)
                for i, (_, row) in enumerate(short_top.iterrows()):
                    with cols_s[i]:
                        with st.container(border=True):
                            st.markdown(f"#### 👑 第{row['順位']}位：{row['銘柄名']}")
                            st.metric("現在値", f"¥{row.get('今日の終値', 0):.1f}")
                            st.metric("総合スコア (相対)", f"{row.get('短期スコア', 0):.1f}")
                            st.metric("絶対上昇確率", f"{row.get('明日の上昇確率', 0):.1f}%")
                            st.caption(f"💡 AIの評価: {row.get('おすすめ理由', '')}")
            
            with st.expander("👉 スキャン対象の全データを詳しく見る (AIの評価順)"):
                st.dataframe(df_rec, hide_index=True, use_container_width=True)
        except Exception as e:
            st.error(f"おすすめデータの読み込みに失敗しました: {e}")

    st.subheader("🔍 あなたの監視リストの手動分析（自己学習・PSI監視）")
    tickers = get_tickers()
    
    if st.button("🚀 監視銘柄の最新分析を実行", type="primary") and tickers:
        with st.spinner('データを一括取得し、Optunaによる自己学習とドリフト検知を実行中...'):
            stock_data_dict = fetch_and_prepare_all_data(tickers)
            
            df_all = []
            for name, data in stock_data_dict.items():
                data['Ticker'] = name
                df_all.append(data)
                
            if df_all:
                df_panel = pd.concat(df_all).sort_index()
                df_panel = df_panel.dropna(subset=EXTENDED_FEATURES + ['Target_Return', 'Target_Class'])
                
                if len(df_panel) > 100:
                    df_panel['Target_Rank'] = df_panel.groupby(level=0)['Target_Return'].transform(
                        lambda x: ((x.rank(method='first') - 1) / max(1, len(x) - 1) * min(4, len(x) - 1)).astype(int)
                    )

                    latest_date = df_panel.index.max()
                    train_df = df_panel[df_panel.index < latest_date]
                    latest_df = df_panel[df_panel.index == latest_date]
                    
                    X_train_raw = train_df[EXTENDED_FEATURES]
                    y_train_rank = train_df['Target_Rank']
                    y_train_class = train_df['Target_Class']
                    
                    scaler = RobustScaler()
                    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), index=X_train_raw.index, columns=EXTENDED_FEATURES)
                    X_latest_scaled = pd.DataFrame(scaler.transform(latest_df[EXTENDED_FEATURES]), index=latest_df.index, columns=EXTENDED_FEATURES)
                    
                    group_train = train_df.groupby(level=0).size().values
                    
                    fs_model = lgb.LGBMRanker(n_estimators=50, random_state=42, verbose=-1)
                    fs_model.fit(X_train_scaled, y_train_rank, group=group_train)
                    importance = pd.Series(fs_model.feature_importances_, index=EXTENDED_FEATURES).sort_values(ascending=False)
                    top_k = min(int(len(EXTENDED_FEATURES) * 0.8), len(EXTENDED_FEATURES))
                    selected_features = importance.head(top_k).index.tolist()
                    
                    X_train_sel = X_train_scaled[selected_features]
                    X_latest_sel = X_latest_scaled[selected_features]
                    
                    def objective(trial):
                        params = {
                            'n_estimators': trial.suggest_int('n_estimators', 50, 100),
                            'max_depth': trial.suggest_int('max_depth', 3, 6),
                            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                            'num_leaves': trial.suggest_int('num_leaves', 10, 30),
                            'random_state': 42,
                            'verbose': -1
                        }
                        dates = train_df.index.unique()
                        split_idx = int(len(dates) * 0.8)
                        split_date = dates[split_idx]
                        mask_train = train_df.index < split_date
                        mask_val = train_df.index >= split_date
                        
                        X_t, y_t = X_train_sel[mask_train], y_train_rank[mask_train]
                        g_t = train_df[mask_train].groupby(level=0).size().values
                        X_v, y_v = X_train_sel[mask_val], y_train_rank[mask_val]
                        g_v = train_df[mask_val].groupby(level=0).size().values
                        
                        if len(g_t) == 0 or len(g_v) == 0: return 0.0
                        
                        model = lgb.LGBMRanker(**params)
                        model.fit(X_t, y_t, group=g_t, eval_set=[(X_v, y_v)], eval_group=[g_v], callbacks=[lgb.early_stopping(10, verbose=False)])
                        return model.best_score_['valid_0']['ndcg@1']
                        
                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=5)
                    
                    best_params = study.best_params
                    best_params['random_state'] = 42
                    best_params['verbose'] = -1
                    
                    ranker = lgb.LGBMRanker(**best_params)
                    ranker.fit(X_train_sel, y_train_rank, group=group_train)
                    latest_df['Ranking_Score'] = ranker.predict(X_latest_sel)
                    
                    clf = lgb.LGBMClassifier(**best_params)
                    clf.fit(X_train_sel, y_train_class)
                    latest_df['Prob_Up'] = clf.predict_proba(X_latest_sel)[:, 1]
                    
                    train_probs = clf.predict_proba(X_train_sel)[:, 1]
                    recent_mask = train_df.index >= (train_df.index.max() - pd.Timedelta(days=30))
                    psi_value = 0
                    if recent_mask.sum() > 0:
                        recent_probs = clf.predict_proba(X_train_sel[recent_mask])[:, 1]
                        psi_value = calculate_psi(train_probs, recent_probs)
                    
                    min_s = latest_df['Ranking_Score'].min()
                    max_s = latest_df['Ranking_Score'].max()
                    if max_s > min_s:
                        latest_df['Normalized_Score'] = (latest_df['Ranking_Score'] - min_s) / (max_s - min_s) * 100
                    else:
                        latest_df['Normalized_Score'] = 50.0
                        
                    latest_df = latest_df.sort_values(by='Ranking_Score', ascending=False)
                    
                    results = []
                    for i, (idx, row) in enumerate(latest_df.iterrows()):
                        res_dict = {
                            "順位": int(i + 1),
                            "銘柄名": row['Ticker'],
                            "現在価格": float(row['Close']),
                            "相対スコア(ランキング)": float(row['Normalized_Score']),
                            "絶対上昇確率(メタ確信度)": float(row['Prob_Up'] * 100)
                        }
                        results.append(res_dict)

            if results:
                df_res = pd.DataFrame(results)
                st.success(f"✅ 自己学習(Optuna)と特徴量選択が完了しました！ (最適パラメータ: {best_params})")
                
                if psi_value > 0.2:
                    st.error(f"⚠️ 警告: PSIスコアが {psi_value:.4f} と異常値を示しています。相場環境が激変しているため、AIの予測精度が低下している可能性があります！")
                elif psi_value > 0.1:
                    st.warning(f"ℹ️ 注意: PSIスコアが {psi_value:.4f} です。相場の傾向に変化の兆しがあります。")
                else:
                    st.info(f"📊 PSIスコア (コンセプトドリフト): {psi_value:.4f} -> 相場は正常・安定しています。")
                
                st.write("### 📊 本日の相場に最適化された相対ランキング")
                cfg = {
                    "順位": st.column_config.NumberColumn(format="%d位"),
                    "現在価格": st.column_config.NumberColumn(format="¥%.1f"), 
                    "相対スコア(ランキング)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
                    "絶対上昇確率(メタ確信度)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%")
                }
                # 🔥 修正: heightを212px（ヘッダー1行＋データ5行の計6行分）に固定し、コンパクトに表示する
                st.dataframe(df_res, column_config=cfg, hide_index=True, use_container_width=True, height=212)
                
                # 🔥 追加: 表の下に「透明な余白（改行3つ分）」を強制的に挿入し、見切れを完全に防ぐ
                st.markdown("<br><br><br>", unsafe_allow_html=True)
            else:
                st.warning("データが不足しているため分析できませんでした。")

with tab2:
    st.write("### 📈 AI予測の過去の推移")
    csv_file = 'prediction_history.csv'
    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
        try:
            df_history = pd.read_csv(csv_file)
            if not df_history.empty:
                stock_list = df_history['銘柄名'].unique()
                selected_stock = st.selectbox("銘柄を選択してください", stock_list, key="tab2_stock")
                df_s = df_history[df_history['銘柄名'] == selected_stock].copy()
                df_s['Date'] = pd.to_datetime(df_s['Date'])
                df_s = df_s.sort_values('Date').reset_index(drop=True)
                
                available_cols = ['明日の上昇確率', '短期スコア', 'メタ確信度']
                cols_to_plot = [c for c in available_cols if c in df_s.columns]
                        
                if cols_to_plot:
                    fig = px.line(df_s, x='Date', y=cols_to_plot, title=f"{selected_stock} の予測スコア推移", markers=True)
                    fig.update_layout(yaxis_range=[0, 100], yaxis_title="スコア / 確率 (%)")
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                st.write("#### 💸 予測履歴からの「仮想損益」シミュレーション")
                tickers_dict = get_tickers()
                selected_ticker = tickers_dict.get(selected_stock, "")
                default_lot = 100 if selected_ticker.endswith(".T") else 1
                
                col_sim1, col_sim2, col_sim3 = st.columns(3)
                with col_sim1: sim_threshold = st.slider("買い条件(スコア)", min_value=50, max_value=90, value=55, step=1, key="tab2_sim")
                with col_sim2: sim_initial_cash = st.number_input("初期資金", value=1000000, step=100000, key="tab2_cash")
                with col_sim3: sim_lot_size = st.number_input("1回の購入株数", value=default_lot, step=1, key="tab2_lot")
                
                cash = sim_initial_cash
                sim_history = []
                
                for i in range(len(df_s)):
                    current_date = df_s.iloc[i]['Date']
                    current_price = df_s.iloc[i]['今日の終値']
                    
                    current_prob = 0
                    if '短期スコア' in df_s.columns: current_prob = df_s.iloc[i]['短期スコア']
                    elif '明日の上昇確率' in df_s.columns: current_prob = df_s.iloc[i]['明日の上昇確率']
                    
                    profit = 0
                    executed = False
                    
                    if i < len(df_s) - 1:
                        next_price = df_s.iloc[i+1]['今日の終値']
                        required_cash = current_price * sim_lot_size
                        if current_prob >= sim_threshold and cash >= required_cash:
                            profit = (next_price - current_price) * sim_lot_size
                            cash += profit
                            executed = True
                            
                    sim_history.append({
                        'Date': current_date, '今日の終値': current_price, '判断スコア': current_prob,
                        'シグナル': "買い" if executed else "-", '損益(円)': profit if executed else 0, '仮想累積資産(円)': cash
                    })
                
                df_sim = pd.DataFrame(sim_history)
                fig2 = px.line(df_sim, x='Date', y='仮想累積資産(円)', title=f"{selected_stock} の仮想累積資産推移", markers=True)
                fig2.add_hline(y=sim_initial_cash, line_dash="dash", line_color="gray", annotation_text="初期資金")
                st.plotly_chart(fig2, use_container_width=True)
                
        except Exception as e:
            st.error(f"履歴データの読み込みエラー: {e}")

with tab3:
    st.write("### 📊 単一銘柄ポテンシャル検証 (LGBMClassifier)")
    st.write("※ここではRankerではなく、従来通りその銘柄が単独で利益を出せるかのシビアな検証を行います。")
    tickers_dict = get_tickers()
    if not tickers_dict:
        st.warning("サイドバーで監視リストを設定してください。")
    else:
        col1, col2 = st.columns(2)
        with col1:
            bt_stock_name = st.selectbox("検証する銘柄", list(tickers_dict.keys()), key="bt_stock")
            bt_ticker = tickers_dict[bt_stock_name]
            default_lot = 100 if str(bt_ticker).endswith(".T") else 1
            bt_lot_size = st.number_input("1回の購入株数", value=default_lot, step=1)
            bt_initial_cash = st.number_input("初期資金（円）", value=1000000, step=100000, key="bt_cash")
            bt_threshold = st.slider("買い条件（確率％）", min_value=50, max_value=80, value=55, step=1)
        with col2:
            bt_tp = st.number_input("利確幅（％）", value=5.0, step=1.0) / 100.0
            bt_sl = st.number_input("損切幅（％）", value=5.0, step=1.0) / 100.0
            bt_hold_days = st.number_input("最大保有日数", value=5, step=1)

        @st.cache_data(show_spinner=False)
        def run_purged_walk_forward_cv(ticker_name, hold_days, _tickers_dict):
            stock_data_dict = fetch_and_prepare_all_data(_tickers_dict)
            data = stock_data_dict.get(ticker_name)
            if data is None: return None, None
            
            data_features = data.dropna(subset=EXTENDED_FEATURES + ['Target_Class'])
            if len(data_features) <= 300: return None, None
            
            n_splits = 4 
            purge_days = int(hold_days) + 1 
            test_probs_all = pd.Series(dtype=float)
            test_data_all = pd.DataFrame()
            
            indices = np.arange(len(data_features))
            fold_size = len(data_features) // (n_splits + 1)
            
            for i in range(n_splits):
                train_end = (i + 1) * fold_size
                test_start = train_end + purge_days
                test_end = test_start + fold_size if i < n_splits - 1 else len(data_features)
                if test_start >= len(data_features): break
                
                train_data = data_features.iloc[indices[:train_end]]
                test_data = data_features.iloc[indices[test_start:test_end]]
                
                X_train = train_data[EXTENDED_FEATURES]
                y_train = train_data['Target_Class']
                X_test = test_data[EXTENDED_FEATURES]
                
                scaler = RobustScaler()
                X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=EXTENDED_FEATURES)
                X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=EXTENDED_FEATURES)
                
                clf = lgb.LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
                clf.fit(X_train_scaled, y_train)
                
                probs = clf.predict_proba(X_test_scaled)[:, 1]
                test_probs_all = pd.concat([test_probs_all, pd.Series(probs, index=test_data.index)])
                test_data_all = pd.concat([test_data_all, test_data])
                
            return test_data_all, test_probs_all

        if st.button("🔄 厳密な実戦シミュレーションを実行 (Purged CV)", type="primary"):
            with st.spinner('過去5年間を通じた高度特徴量でのウォークフォワード検証を実行中...'):
                test_data, test_probs = run_purged_walk_forward_cv(bt_stock_name, bt_hold_days, tickers_dict)

            if test_data is not None and test_probs is not None:
                cash = bt_initial_cash
                position = 0; entry_price = 0; days_held = 0
                trades = []; history = []
                
                initial_price = test_data['Close'].iloc[0]
                bh_shares = bt_lot_size if bt_initial_cash >= (initial_price * bt_lot_size) else 0
                bh_cash = bt_initial_cash - (bh_shares * initial_price)
                
                for i in range(len(test_data)):
                    current_date = test_data.index[i]
                    current_price = test_data['Close'].iloc[i]
                    current_prob = test_probs[i]
                    
                    if position > 0:
                        days_held += 1
                        ret = (current_price - entry_price) / entry_price
                        if ret >= bt_tp or ret <= -bt_sl or days_held >= bt_hold_days or i == len(test_data)-1:
                            profit = position * (current_price - entry_price)
                            cash += position * current_price
                            trades.append({'決済日': current_date.strftime('%Y-%m-%d'), '損益(円)': int(profit)})
                            position = 0; entry_price = 0; days_held = 0
                    
                    if position == 0 and current_prob >= (bt_threshold / 100.0) and i < len(test_data)-1:
                        required_cash = current_price * bt_lot_size
                        if cash >= required_cash:
                            position = bt_lot_size; entry_price = current_price
                            cash -= required_cash; days_held = 0
                    
                    equity = cash + (position * current_price)
                    bh_equity = bh_cash + (bh_shares * current_price)
                    history.append({'Date': current_date, 'AI戦略の資産': equity, '放置(ガチホ)の資産': bh_equity})
                
                history_df = pd.DataFrame(history)
                trades_df = pd.DataFrame(trades)
                final_equity = history_df.iloc[-1]['AI戦略の資産']
                total_profit = final_equity - bt_initial_cash
                profit_pct = (total_profit / bt_initial_cash) * 100
                bh_profit_pct = ((history_df.iloc[-1]['放置(ガチホ)の資産'] - bt_initial_cash) / bt_initial_cash) * 100
                win_rate = len(trades_df[trades_df['損益(円)'] > 0]) / len(trades_df) * 100 if not trades_df.empty else 0
                    
                st.write("---")
                if trades_df.empty and bh_shares == 0:
                    st.warning(f"⚠️ 初期資金が足りず、一度も購入できませんでした。")
                
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("AI運用 最終資金", f"¥{int(final_equity):,}")
                mcol2.metric("AI運用 総損益", f"¥{int(total_profit):,}", f"{profit_pct:.1f}%")
                mcol3.metric("勝率", f"{win_rate:.1f}%")
                mcol4.metric("取引回数", f"{len(trades_df)}回")
                st.plotly_chart(px.line(history_df, x='Date', y=['AI戦略の資産', '放置(ガチホ)の資産'], title="資産推移の比較（エクイティカーブ）"), use_container_width=True)