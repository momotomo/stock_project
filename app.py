import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
import os
import plotly.express as px
from github import Github

# ページ設定
st.set_page_config(layout="wide", page_title="AI株価スクリーニング")

# --- Secrets読込バックアップ機能 ---
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

# --- 分析・特徴量作成関数（共通化） ---
@st.cache_data(show_spinner=False)
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
    except Exception:
        return pd.DataFrame()

def get_stock_features(ticker, macro_returns):
    data = yf.download(ticker, period="5y", progress=False)
    if data.empty: return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.loc[:, ~data.columns.duplicated()].copy()
    data.index = pd.to_datetime(data.index).map(lambda x: x.replace(tzinfo=None).normalize())
    
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
    return data

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

tab1, tab2, tab3 = st.tabs(["🔍 今日の分析 ＆ おすすめ", "📈 予測推移", "📊 バックテスト (検証)"])

# --- サイドバー設定 ---
st.sidebar.header("⚙️ 分析銘柄の設定")
TICKERS_FILE = "tickers.txt"

def load_tickers_text():
    if os.path.exists(TICKERS_FILE):
        with open(TICKERS_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return "トヨタ自動車, 7203.T\n三菱UFJ, 8306.T\nソニーG, 6758.T\nソフトバンクG, 9984.T\n任天堂, 7974.T"

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

tickers = {}
for line in ticker_input.strip().split('\n'):
    if ',' in line:
        name, ticker = line.split(',', 1)
        tickers[name.strip()] = ticker.strip()

# ==========================================
# タブ1：今日の分析 ＆ AIおすすめ
# ==========================================
with tab1:
    # --- 🌟 AIおすすめ銘柄セクション ---
    st.subheader("🌟 AIが選ぶ本日のおすすめ銘柄 (自動更新)")
    st.write("日本の大型優良株30銘柄のチャートをAIが毎朝自動でスキャンし、目的に合わせたトップ銘柄を厳選しています。")
    
    rec_file = 'recommendations.csv'
    if os.path.exists(rec_file) and os.path.getsize(rec_file) > 0:
        try:
            df_rec = pd.read_csv(rec_file)
            
            st.markdown("### ⚡ 短期取引向け（明日〜2週間）トップ3")
            if "短期スコア" in df_rec.columns:
                short_top = df_rec.sort_values(by="短期スコア", ascending=False).head(3)
                cols_s = st.columns(3)
                for i, (_, row) in enumerate(short_top.iterrows()):
                    with cols_s[i]:
                        st.info(f"**第{i+1}位：{row['銘柄名']}**\n\n現在値: ¥{row.get('今日の終値', '---')}\n\n短期期待値: **{row.get('短期スコア', 0):.1f}%**\n\n💡 **AIの評価**: {row.get('おすすめ理由', '')}")
            
            st.markdown("### 🔭 中長期取引向け（1ヶ月〜1年）トップ3")
            if "中長期スコア" in df_rec.columns:
                long_top = df_rec.sort_values(by="中長期スコア", ascending=False).head(3)
                cols_l = st.columns(3)
                for i, (_, row) in enumerate(long_top.iterrows()):
                    with cols_l[i]:
                        st.success(f"**第{i+1}位：{row['銘柄名']}**\n\n現在値: ¥{row.get('今日の終値', '---')}\n\n中長期期待値: **{row.get('中長期スコア', 0):.1f}%**\n\n💡 **AIの評価**: {row.get('おすすめ理由', '')}")
                    
            with st.expander("👉 スキャン対象の全データを詳しく見る"):
                st.dataframe(df_rec, hide_index=True, use_container_width=True)
        except Exception as e:
            st.error(f"おすすめデータの読み込みに失敗しました: {e}")
    else:
        st.info("💡 現在おすすめ銘柄を分析中です。次回の自動バッチ処理が完了するとここに表示されます！")
        
    st.divider()

    # --- 手動分析セクション ---
    st.subheader("🔍 あなたの監視リストの手動分析（全スパン）")
    if st.button("🚀 監視銘柄の最新分析を実行", type="primary") and tickers:
        results = []
        tb_horizons = {'1W': 5, '2W': 10, '1M': 21, '3M': 63, '6M': 126, '1Y': 252}
        pt_sl = {'1W': 0.03, '2W': 0.05, '1M': 0.10, '3M': 0.20, '6M': 0.30, '1Y': 0.50}

        with st.spinner('データを取得し、AIモデルを学習しています...'):
            macro_returns = get_macro_data()
            for name, ticker in tickers.items():
                data = get_stock_features(ticker, macro_returns)
                if data is None: continue
                
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
                
                res_dict = {
                    "銘柄名": name,
                    "現在価格": float(current_price),
                    "予測価格": float(pred_price),
                    "明日の上昇確率": float(prob_up * 100)
                }
                for k in tb_horizons.keys():
                    res_dict[f"{k} 利確(>{int(pt_sl[k]*100)}%)"] = float(tb_results[k] * 100)
                results.append(res_dict)

        if results:
            df_res = pd.DataFrame(results).sort_values("明日の上昇確率", ascending=False)
            st.write("### 📊 本日の分析結果（全スパン）")
            
            cfg = {"現在価格": st.column_config.NumberColumn(format="¥%.1f"), "予測価格": st.column_config.NumberColumn(format="¥%.1f")}
            for col in df_res.columns:
                if "確率" in col or "利確" in col:
                    cfg[col] = st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%")
            
            st.dataframe(df_res, column_config=cfg, hide_index=True, use_container_width=True)

# ==========================================
# タブ2：過去の予測推移
# ==========================================
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
                
                # 存在しているスパンのカラムを動的に取得
                available_cols = ['明日の上昇確率']
                for col in ['1W 利確(>3%)', '2W 利確(>5%)', '1M 利確(>10%)', '3M 利確(>20%)', '6M 利確(>30%)', '1Y 利確(>50%)']:
                    if col in df_s.columns:
                        available_cols.append(col)
                        
                fig = px.line(df_s, x='Date', y=available_cols, title=f"{selected_stock} の予測確率推移", markers=True)
                fig.update_layout(yaxis_range=[0, 100], yaxis_title="確率 (%)")
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                st.write("#### 💸 予測履歴からの「仮想損益」シミュレーション")
                selected_ticker = tickers.get(selected_stock, "")
                default_lot = 100 if selected_ticker.endswith(".T") else 1
                
                col_sim1, col_sim2, col_sim3 = st.columns(3)
                with col_sim1:
                    sim_threshold = st.slider("買い条件（明日の上昇確率 ％以上）", min_value=50, max_value=90, value=55, step=1, key="tab2_sim")
                with col_sim2:
                    sim_initial_cash = st.number_input("初期資金（円）", value=1000000, step=100000, key="tab2_cash")
                with col_sim3:
                    sim_lot_size = st.number_input("1回の購入株数（単元株数）", value=default_lot, step=1, key="tab2_lot")
                
                cash = sim_initial_cash
                sim_history = []
                
                for i in range(len(df_s)):
                    current_date = df_s.iloc[i]['Date']
                    current_price = df_s.iloc[i]['今日の終値']
                    current_prob = df_s.iloc[i]['明日の上昇確率']
                    
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
                        'Date': current_date,
                        '今日の終値': current_price,
                        '明日の上昇確率': current_prob,
                        'シグナル': "買い" if executed else "-",
                        '損益(円)': profit if executed else 0,
                        '仮想累積資産(円)': cash
                    })
                
                df_sim = pd.DataFrame(sim_history)
                
                fig2 = px.line(df_sim, x='Date', y='仮想累積資産(円)', title=f"{selected_stock} の仮想累積資産推移 (円)", markers=True)
                fig2.add_hline(y=sim_initial_cash, line_dash="dash", line_color="gray", annotation_text="初期資金")
                st.plotly_chart(fig2, use_container_width=True)
                
                st.write("#### データ詳細（最新順）")
                st.dataframe(df_sim.sort_values('Date', ascending=False), hide_index=True, use_container_width=True)
        except Exception as e:
            st.error(f"履歴データの読み込みエラー: {e}")
    else:
        st.info("まだ履歴データがありません。自動実行が開始されるまでお待ちください。")

# ==========================================
# タブ3：バックテスト (検証)
# ==========================================
with tab3:
    st.write("### 📊 AIシグナルによる運用シミュレーション")
    
    if not tickers:
        st.warning("サイドバーで監視リストを設定してください。")
    else:
        col1, col2 = st.columns(2)
        with col1:
            bt_stock_name = st.selectbox("検証する銘柄", list(tickers.keys()), key="bt_stock")
            bt_ticker = tickers[bt_stock_name]
            default_lot = 100 if str(bt_ticker).endswith(".T") else 1
            bt_lot_size = st.number_input("1回の購入株数（単元株数）", value=default_lot, step=1)
            bt_initial_cash = st.number_input("初期資金（円）", value=1000000, step=100000, key="bt_cash")
            bt_threshold = st.slider("買い条件（明日の上昇確率が何％以上で買うか）", min_value=50, max_value=80, value=55, step=1)
            
        with col2:
            bt_tp = st.number_input("利確幅（％）", value=5.0, step=1.0) / 100.0
            bt_sl = st.number_input("損切幅（％）", value=5.0, step=1.0) / 100.0
            bt_hold_days = st.number_input("最大保有日数（強制決済）", value=5, step=1)

        @st.cache_data(show_spinner=False)
        def run_ml_prediction_for_bt(ticker):
            macro_returns = get_macro_data()
            data = get_stock_features(ticker, macro_returns)
            if data is None: return None, None
            
            features = ['Log_Return', 'Disparity_5', 'Disparity_25', 'SMA_Cross', 'RSI_14', 
                        'BB_PctB', 'BB_Bandwidth', 'MACD_Norm', 'ATR', 'OBV_Ret', 'USDJPY_Ret', 'SP500_Ret']
            data_features = data.dropna(subset=features + ['Target_Class'])
            
            if len(data_features) <= 300: return None, None
            
            test_size = 250
            train_data = data_features.iloc[:-test_size]
            test_data = data_features.iloc[-test_size:]
            
            X_train = train_data[features]
            y_train = train_data['Target_Class']
            
            scaler = RobustScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=features)
            X_test_scaled = pd.DataFrame(scaler.transform(test_data[features]), index=test_data.index, columns=features)
            
            clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            clf.fit(X_train_scaled, y_train)
            test_probs = clf.predict_proba(X_test_scaled)[:, 1]
            
            return test_data, test_probs

        if st.button("🔄 シミュレーションを実行", type="primary"):
            with st.spinner('過去のチャートを分析中...'):
                test_data, test_probs = run_ml_prediction_for_bt(bt_ticker)

            if test_data is not None and test_probs is not None:
                cash = bt_initial_cash
                position = 0
                entry_price = 0
                days_held = 0
                trades = []
                history = []
                
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
                            trades.append({
                                '決済日': current_date.strftime('%Y-%m-%d'),
                                '損益(円)': int(profit),
                            })
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
                    st.warning(f"⚠️ 初期資金（{bt_initial_cash}円）が足りず、一度も購入できませんでした。")
                
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("AI運用 最終資金", f"¥{int(final_equity):,}")
                mcol2.metric("AI運用 総損益", f"¥{int(total_profit):,}", f"{profit_pct:.1f}%")
                mcol3.metric("勝率", f"{win_rate:.1f}%")
                mcol4.metric("取引回数", f"{len(trades_df)}回")
                
                st.plotly_chart(px.line(history_df, x='Date', y=['AI戦略の資産', '放置(ガチホ)の資産'], title="資産推移の比較（エクイティカーブ）"), use_container_width=True)
            else:
                st.warning("データ不足のためバックテストを実行できません。")