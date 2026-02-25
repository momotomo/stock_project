import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import os
import plotly.express as px
from github import Github

# ページ設定
st.set_page_config(layout="wide", page_title="AI株価スクリーニング")

# --- Secrets読込バックアップ機能（綴りミスや環境差への対策） ---
def get_secret(key):
    """通常のSecretsとファイル直接読込の両方からキーを探す"""
    # 1. Streamlit標準のSecrets（クラウド環境および正常なローカル環境用）
    if key in st.secrets:
        return st.secrets[key]
    
    # 2. 手動でのファイル検索（ローカルで自動読込が認識されない場合のバックアップ）
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

# --- トリプルバリア法のラベリング関数 ---
@st.cache_data
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

st.title("🚀 AI株価スクリーニングダッシュボード")

tab1, tab2 = st.tabs(["🔍 今日の分析を実行", "📈 過去の予測推移を見る"])

with tab1:
    st.write("短期の予測に加え、中長期スパンでの「損切り前に利確ラインに到達する確率」を複数のAIモデルで並行予測します。")

    # --- サイドバー設定 ---
    st.sidebar.header("⚙️ 分析銘柄の設定")
    TICKERS_FILE = "tickers.txt"
    
    def load_tickers_text():
        if os.path.exists(TICKERS_FILE):
            with open(TICKERS_FILE, "r", encoding="utf-8") as f:
                return f.read()
        return "トヨタ自動車, 7203.T\n三菱UFJ, 8306.T\nソニーG, 6758.T\nソフトバンクG, 9984.T\n任天堂, 7974.T"

    ticker_input = st.sidebar.text_area(
        "監視リスト", 
        value=load_tickers_text(), 
        height=300,
        help="1行に1銘柄ずつ「銘柄名, ティッカー」の形式で入力してください。"
    )

    # 監視リストの保存・同期ボタン
    if st.sidebar.button("💾 監視リストを保存・同期", use_container_width=True):
        # ローカルの tickers.txt を更新
        with open(TICKERS_FILE, "w", encoding="utf-8") as f:
            f.write(ticker_input)
        
        # GitHub同期を試行（手動読込関数 get_secret を使用）
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
            st.sidebar.warning("✅ ローカルに保存しました（GitHub同期設定が未認識です）")

    tickers = {}
    for line in ticker_input.strip().split('\n'):
        line = line.strip()
        if ',' in line:
            name, ticker = line.split(',', 1)
            tickers[name.strip()] = ticker.strip()

    # --- 分析実行 ---
    if st.button("🔍 プロ仕様の分析を実行する", type="primary") and tickers:
        results = []
        with st.spinner('最新データを取得し、AIモデルを学習しています...'):
            # マクロデータ取得
            macro_tickers = {"USDJPY=X": "USDJPY_Ret", "^GSPC": "SP500_Ret"}
            try:
                macro_df = yf.download(list(macro_tickers.keys()), period="5y", progress=False)
                if isinstance(macro_df.columns, pd.MultiIndex):
                    macro_data = macro_df.xs('Close', level=0, axis=1) if 'Close' in macro_df.columns.levels[0] else macro_df
                else:
                    macro_data = macro_df['Close']
                macro_returns = np.log(macro_data / macro_data.shift(1)).rename(columns=macro_tickers)
                macro_returns.index = pd.to_datetime(macro_returns.index).map(lambda x: x.replace(tzinfo=None).normalize())
            except Exception:
                macro_returns = pd.DataFrame()

            for name, ticker in tickers.items():
                try:
                    data = yf.download(ticker, period="5y", progress=False)
                    if data.empty: continue
                    
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
                    
                    # 予測ターゲット
                    data['Target_Class'] = (data['Close'].shift(-1) > data['Close']).astype(int)
                    data['Target_Price'] = data['Close'].shift(-1)
                    
                    features = ['Log_Return', 'Disparity_5', 'Disparity_25', 'SMA_Cross', 'RSI_14', 
                                'BB_PctB', 'BB_Bandwidth', 'MACD_Norm', 'ATR', 'OBV_Ret', 'USDJPY_Ret', 'SP500_Ret']
                    
                    data_features = data.dropna(subset=features)
                    if len(data_features) <= 100: continue
                    
                    latest_data = data_features.iloc[[-1]]
                    current_price = latest_data['Close'].values[0]
                    historical_data = data_features.dropna(subset=['Target_Class', 'Target_Price'])
                    
                    if len(historical_data) <= 100: continue
                    
                    X_raw = historical_data[features]
                    scaler = RobustScaler()
                    X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), index=X_raw.index, columns=features)
                    latest_X_scaled = pd.DataFrame(scaler.transform(latest_data[features]), index=latest_data.index, columns=features)
                    
                    # 機械学習実行
                    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                    clf.fit(X_scaled, historical_data['Target_Class'])
                    prob_up = clf.predict_proba(latest_X_scaled)[0][1]
                    
                    reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
                    reg.fit(X_scaled, historical_data['Target_Price'])
                    pred_price = reg.predict(latest_X_scaled)[0]
                    
                    # トリプルバリア（中長期利確確率）
                    tb_horizons = {'1W': 5, '2W': 10, '1M': 21, '3M': 63, '6M': 126, '1Y': 252}
                    pt_sl = {'1W':0.03, '2W':0.05, '1M':0.10, '3M':0.20, '6M':0.30, '1Y':0.50}
                    tb_results = {}
                    for h_key, h_days in tb_horizons.items():
                        target_col = f'Target_TB_{h_key}'
                        data[target_col] = calc_triple_barrier(data['Close'], h_days, pt_sl[h_key], pt_sl[h_key])
                        v_idx = data[data[target_col].notna()].index.intersection(X_raw.index)
                        if len(v_idx) > 100:
                            clf_h = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                            clf_h.fit(X_scaled.loc[v_idx], data.loc[v_idx, target_col])
                            cls = list(clf_h.classes_)
                            p1 = clf_h.predict_proba(latest_X_scaled)[0][cls.index(1)] if 1 in cls else 0.0
                            tb_results[f"{h_key}利確"] = float(p1 * 100)
                        else:
                            tb_results[f"{h_key}利確"] = 0.0
                    
                    res = {"銘柄名": name, "現在価格": float(current_price), "予測価格": float(pred_price), "明日上昇率": float(prob_up * 100)}
                    res.update(tb_results)
                    results.append(res)
                except Exception:
                    continue

        if results:
            df_res = pd.DataFrame(results)
            st.write("### 📊 本日のスクリーニング結果")
            st.dataframe(
                df_res,
                column_config={
                    "明日上昇率": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
                    "1W利確": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
                    "1M利確": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
                    "3M利確": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
                    "現在価格": st.column_config.NumberColumn(format="¥%.1f"),
                    "予測価格": st.column_config.NumberColumn(format="¥%.1f"),
                },
                hide_index=True, 
                use_container_width=True
            )
        else:
            st.warning("⚠️ 分析可能なデータがありませんでした。ティッカーシンボルが正しいか確認してください。")

# --- 履歴タブ ---
with tab2:
    st.write("### 📈 AI予測の過去の推移")
    csv_file = 'prediction_history.csv'
    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
        try:
            df_history = pd.read_csv(csv_file)
            if not df_history.empty:
                stock_list = df_history['銘柄名'].unique()
                selected_stock = st.selectbox("銘柄を選択してください", stock_list)
                df_s = df_history[df_history['銘柄名'] == selected_stock].copy()
                df_s['Date'] = pd.to_datetime(df_s['Date'])
                df_s = df_s.sort_values('Date')
                
                # チャートの作成（上昇確率の推移）
                fig = px.line(df_s, x='Date', y=['明日の上昇確率', '1M 利確(>10%)'], 
                              title=f"{selected_stock} の予測確率推移", markers=True)
                fig.update_layout(yaxis_range=[0, 100], yaxis_title="確率 (%)")
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("#### データ詳細（最新順）")
                st.dataframe(df_s.sort_values('Date', ascending=False), hide_index=True, use_container_width=True)
        except Exception as e:
            st.error(f"履歴データの読み込み中にエラーが発生しました: {e}")
    else:
        st.info("まだ履歴データがありません。自動実行が開始されるまでお待ちください。")