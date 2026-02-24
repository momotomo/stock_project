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

# ページ全体を広く使う設定
st.set_page_config(layout="wide", page_title="AI株価スクリーニング")

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

st.title("🚀 AI株価スクリーニングダッシュボード (中長期トリプルバリア版)")

tab1, tab2 = st.tabs(["🔍 今日の分析を実行", "📈 過去の予測推移を見る"])

with tab1:
    st.write("短期の予測に加え、中長期スパンでの「損切り前に利確ラインに到達する確率」を複数のAIモデルで並行予測します。")

    with st.expander("🧠 AIが学習に使用している特徴量と「トリプルバリア法」について"):
        st.markdown("""
        ### 📊 使用している特徴量（絶対価格を排除した相対指標）
        1. **Log_Return**: 対数収益率　2. **Disparity_5/25**: SMA乖離率　3. **SMA_Cross**: 移動平均クロス
        4. **RSI_14**: オシレーター　5. **BB_PctB / Bandwidth**: ボリンジャーバンド情報
        6. **MACD_Norm**: 正規化MACD　7. **ATR**: ボラティリティ　8. **OBV_Ret**: 出来高フロー
        9. **USDJPY_Ret / SP500_Ret**: ドル円・S&P500の変動（マクロ指標）
        """)

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
        height=250,
        help="カンマ区切りで銘柄名とYahoo Financeのティッカーを入力してください。"
    )

    if st.sidebar.button("💾 監視リストを保存・同期"):
        with open(TICKERS_FILE, "w", encoding="utf-8") as f:
            f.write(ticker_input)
        if "GITHUB_TOKEN" in st.secrets and "GITHUB_REPO" in st.secrets:
            try:
                g = Github(st.secrets["GITHUB_TOKEN"])
                repo = g.get_repo(st.secrets["GITHUB_REPO"])
                try:
                    contents = repo.get_contents(TICKERS_FILE)
                    repo.update_file(contents.path, "Web画面から更新", ticker_input, contents.sha)
                    st.sidebar.success("✅ GitHubに同期しました！")
                except:
                    repo.create_file(TICKERS_FILE, "新規作成", ticker_input)
                    st.sidebar.success("✅ GitHubに新規作成・同期しました！")
            except Exception as e:
                st.sidebar.error(f"GitHub同期エラー: {e}")
        else:
            st.sidebar.success("✅ ローカルに保存しました！")

    tickers = {}
    for line in ticker_input.strip().split('\n'):
        line = line.strip()
        if ',' in line:
            name, ticker = line.split(',', 1)
            tickers[name.strip()] = ticker.strip()

    if not tickers:
        st.warning("👈 左側のサイドバーから分析する銘柄を少なくとも1つ入力してください。")

    if st.button("🔍 プロ仕様の分析を実行する", type="primary") and tickers:
        results = []
        debug_logs = [] # 🌟 原因究明用のログ記録
        
        with st.spinner('データ取得・AI並行学習中...（最大1〜2分かかります）'):
            macro_tickers = {"USDJPY=X": "USDJPY_Ret", "^GSPC": "SP500_Ret"}
            try:
                macro_df = yf.download(list(macro_tickers.keys()), period="5y", progress=False)
                if isinstance(macro_df.columns, pd.MultiIndex):
                    macro_data = macro_df.xs('Close', level=0, axis=1) if 'Close' in macro_df.columns.levels[0] else macro_df
                else:
                    macro_data = macro_df['Close'] if 'Close' in macro_df else macro_df
                
                macro_returns = np.log(macro_data / macro_data.shift(1)).rename(columns=macro_tickers)
                macro_returns.index = pd.to_datetime(macro_returns.index).map(lambda x: x.replace(tzinfo=None).normalize())
            except Exception as e:
                macro_returns = pd.DataFrame()
                debug_logs.append(f"⚠️ マクロ指標の取得に失敗しました。例外: {e}")
            
            for name, ticker in tickers.items():
                debug_logs.append(f"--- 銘柄: {name} ({ticker}) ---")
                
                try:
                    stock_info = yf.Ticker(ticker).info
                    per = stock_info.get('trailingPE', None) 
                except:
                    per = None
                    
                data = yf.download(ticker, period="5y", progress=False)
                if data.empty:
                    debug_logs.append(f"❌ yfinanceがデータを返しませんでした（ティッカーが間違っているか、通信エラー）。")
                    continue
                    
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data = data.loc[:, ~data.columns.duplicated()].copy()
                data.index = pd.to_datetime(data.index).map(lambda x: x.replace(tzinfo=None).normalize())
                    
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
                
                # 無限大（inf）を取り除きNaNにする（エラー防止）
                data.replace([np.inf, -np.inf], np.nan, inplace=True)
                
                # 🌟 強制マクロ結合（道連れエラー防止）
                if not macro_returns.empty:
                    data = data.join(macro_returns)
                for col in ['USDJPY_Ret', 'SP500_Ret']:
                    if col not in data.columns:
                        data[col] = 0.0 # 取得できなかった場合は変動0として扱う
                        debug_logs.append(f"⚠️ {col} の結合に失敗したため、0で補完しました。")
                        
                data[['USDJPY_Ret', 'SP500_Ret']] = data[['USDJPY_Ret', 'SP500_Ret']].ffill().fillna(0)
                
                data['Target_Class'] = (data['Close'].shift(-1) > data['Close']).astype(int)
                data['Target_Price'] = data['Close'].shift(-1)
                
                features = ['Log_Return', 'Disparity_5', 'Disparity_25', 'SMA_Cross', 'RSI_14', 
                            'BB_PctB', 'BB_Bandwidth', 'MACD_Norm', 'ATR', 'OBV_Ret', 'USDJPY_Ret', 'SP500_Ret']
                
                # 🌟 修正：明日の予測用に「今日のデータ」を確保してからターゲット（答え）の欠損を消す
                data_features = data.dropna(subset=features)
                if len(data_features) <= 100:
                    debug_logs.append(f"❌ 有効な特徴量データが不足しています（件数: {len(data_features)}）。")
                    continue
                    
                latest_data = data_features.iloc[[-1]] # 今日の最新データ（明日の予測に使用）
                current_price = latest_data['Close'].values[0]
                
                historical_data = data_features.dropna(subset=['Target_Class', 'Target_Price'])
                if len(historical_data) <= 100:
                    debug_logs.append(f"❌ AI学習用の過去データが不足しています（件数: {len(historical_data)}）。")
                    continue
                
                debug_logs.append(f"✅ 学習データ準備完了（{len(historical_data)}件）")
                
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
                    tb_results[h_params['name']] = float(prob_1 * 100)
                
                per_str = f"{round(per, 1)}倍" if pd.notna(per) else "-"
                result_dict = {
                    "銘柄名": name,
                    "WF平均正解率(明日)": float(wf_mean_accuracy * 100),
                    "PER(割安)": per_str,
                    "今日の終値": float(current_price),
                    "明日の予測値": float(pred_price),
                    "明日の上昇確率": float(prob_up * 100)
                }
                result_dict.update(tb_results)
                results.append(result_dict)
            
        # 結果の描画（スピナー外）
        if results:
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values(by="1M 利確(>10%)", ascending=False).reset_index(drop=True)
            st.divider()
            st.write("### 📊 本日のスクリーニング結果")
            st.dataframe(
                    df_results,
                    column_config={
                        "銘柄名": st.column_config.TextColumn("銘柄名"),
                        "WF平均正解率(明日)": st.column_config.ProgressColumn("WF正解率", min_value=0, max_value=100, format="%.1f%%"),
                        "PER(割安)": st.column_config.TextColumn("PER"),
                        "今日の終値": st.column_config.NumberColumn("終値", format="¥ %.1f"),
                        "明日の予測値": st.column_config.NumberColumn("予測値", format="¥ %.1f"),
                        "明日の上昇確率": st.column_config.ProgressColumn("明日上昇", min_value=0, max_value=100, format="%.1f%%"),
                        "1W 利確(>3%)": st.column_config.ProgressColumn("1W(>3%)", min_value=0, max_value=100, format="%.1f%%"),
                        "2W 利確(>5%)": st.column_config.ProgressColumn("2W(>5%)", min_value=0, max_value=100, format="%.1f%%"),
                        "1M 利確(>10%)": st.column_config.ProgressColumn("1M(>10%)", min_value=0, max_value=100, format="%.1f%%"),
                        "3M 利確(>20%)": st.column_config.ProgressColumn("3M(>20%)", min_value=0, max_value=100, format="%.1f%%"),
                        "6M 利確(>30%)": st.column_config.ProgressColumn("6M(>30%)", min_value=0, max_value=100, format="%.1f%%"),
                        "1Y 利確(>50%)": st.column_config.ProgressColumn("1Y(>50%)", min_value=0, max_value=100, format="%.1f%%"),
                    },
                    hide_index=True, 
                    use_container_width=True
                )
        else:
            st.warning("⚠️ 全ての銘柄で分析がスキップされ、結果が0件になりました。以下の「エラー原因の調査用データ」を展開して理由を確認してください。")
            with st.expander("🛠 エラー原因の調査用データ（ここを開く）", expanded=True):
                for log in debug_logs:
                    st.write(log)

# --- 🌟 新機能：履歴確認タブ ---
with tab2:
    st.write("### 📈 AI予測の過去の推移")
    st.write("GitHub Actionsによって自動記録された `prediction_history.csv` のデータを表示します。")
    
    csv_file = 'prediction_history.csv'
    
    if os.path.exists(csv_file):
        try:
            df_history = pd.read_csv(csv_file)
            if not df_history.empty:
                selected_stock = st.selectbox("推移を見たい銘柄を選択してください", df_history['銘柄名'].unique())
                df_stock = df_history[df_history['銘柄名'] == selected_stock].copy()
                df_stock['Date'] = pd.to_datetime(df_stock['Date'])
                df_stock = df_stock.sort_values('Date')
                
                fig = px.line(
                    df_stock, 
                    x='Date', 
                    y=['明日の上昇確率', '1W 利確(>3%)', '1M 利確(>10%)'],
                    title=f"{selected_stock} のAI予測確率の推移",
                    labels={'value': '確率 (%)', 'variable': '予測期間'},
                    markers=True
                )
                fig.update_layout(yaxis=dict(range=[0, 100]))
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("#### 📋 記録データ一覧")
                st.dataframe(df_stock.sort_values('Date', ascending=False), hide_index=True, use_container_width=True)
            else:
                st.info("データがまだ記録されていません。")
        except Exception as e:
            st.error(f"エラー: {e}")
    else:
        st.info("データがまだありません。")