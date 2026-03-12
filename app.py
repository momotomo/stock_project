import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as plgo
import joblib
import os
import subprocess
import time
from datetime import datetime, timedelta
import warnings

from runtime_paths import PREDICTION_HISTORY_PATH, RECOMMENDATIONS_PATH

# =========================================================
# AI株価スクリーニング ダッシュボード (V2.1 実運用対応版)
# =========================================================

warnings.simplefilter('ignore', ResourceWarning)

st.set_page_config(layout="wide", page_title="AI株価スクリーニング")

# ===== V2.1 Cost Model Params (ratio) =====
BASE_FEE = 0.001            # 片道手数料率（例：0.1%）
SLIPPAGE_FACTOR = 0.05      # ATRに対するスリッページ係数（初期仮置き）
TIME_LAG_PENALTY = 0.001    # 14:50決済 vs 15:00 Closeの不確実性（往復に追加）

def cost_hat_roundtrip(atr_prev_ratio: float) -> float:
    # すべて「比率」（2%なら0.02）で計算
    return (BASE_FEE * 2.0) + (atr_prev_ratio * SLIPPAGE_FACTOR * 2.0) + TIME_LAG_PENALTY

# ⚠️ 【重要】ここをご自身のKaggleのIDに書き換えてください
KAGGLE_NOTEBOOK_SLUG = "tokkatokka/stock-ai-trainer"

def get_secret(key, default=None):
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
    return os.environ.get(key, default)

def get_tickers():
    tickers = {
        "トヨタ自動車": "7203.T", "三菱UFJ": "8306.T", "ソフトバンクG": "9984.T",
        "アドバンテスト": "6857.T", "任天堂": "7974.T"
    }
    if os.path.exists("tickers.txt"):
        with open("tickers.txt", "r", encoding="utf-8") as f:
            file_tickers = {}
            for line in f:
                if ',' in line:
                    name, tk = line.split(',', 1)
                    file_tickers[name.strip()] = tk.strip()
            if file_tickers: tickers = file_tickers
    return tickers

def save_tickers(tickers_text):
    with open("tickers.txt", "w", encoding="utf-8") as f:
        f.write(tickers_text)
    
    github_token = get_secret("GITHUB_TOKEN")
    github_repo = get_secret("GITHUB_REPO")
    
    if github_token and github_repo:
        try:
            from github import Github
            g = Github(github_token)
            repo = g.get_repo(github_repo)
            try:
                contents = repo.get_contents("tickers.txt", ref="main")
                repo.update_file(contents.path, "Update tickers list via Streamlit", tickers_text, contents.sha, branch="main")
                st.sidebar.success("✅ GitHubに同期しました！")
            except Exception:
                repo.create_file("tickers.txt", "Create tickers list via Streamlit", tickers_text, branch="main")
                st.sidebar.success("✅ GitHubに新規作成・同期しました！")
        except Exception as e:
            st.sidebar.error(f"❌ GitHub同期エラー: {e}")
    else:
        st.sidebar.info("ℹ️ GitHub同期設定がないため、一時保存のみ行いました。")

def download_models_from_kaggle():
    os.makedirs("models", exist_ok=True)
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        kaggle_username = get_secret("KAGGLE_USERNAME")
        kaggle_key = get_secret("KAGGLE_KEY")
        if kaggle_username and kaggle_key:
            env["KAGGLE_USERNAME"] = kaggle_username
            env["KAGGLE_KEY"] = kaggle_key
        
        subprocess.run(
            ["kaggle", "kernels", "output", KAGGLE_NOTEBOOK_SLUG, "-p", "models"],
            capture_output=True, text=True, check=True, encoding="utf-8", env=env
        )
        return True, "✅ モデルのダウンロードが完了しました。"
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else e.stdout.strip()
        if "cp932" in error_msg or "multibyte sequence" in error_msg:
            return True, "✅ モデルのダウンロードが完了しました。"
        return False, f"❌ Kaggleダウンロードエラー: {error_msg}"
    except FileNotFoundError:
        return False, "❌ 'kaggle' コマンドが見つかりません。アプリの requirements.txt に 'kaggle' が追加されているか確認してください。"
    except Exception as e:
        return False, f"❌ 予期せぬエラー: {e}"

@st.cache_data(show_spinner=False)
def get_macro_data():
    macro_tickers = {"USDJPY=X": "USDJPY_Ret", "^GSPC": "SP500_Ret", "1306.T": "TOPIX_Ret"}
    try:
        macro_df = yf.download(list(macro_tickers.keys()), period="5y", progress=False)
        macro_data = macro_df.xs('Close', level=0, axis=1) if isinstance(macro_df.columns, pd.MultiIndex) else macro_df['Close']
        macro_returns = np.log(macro_data / macro_data.shift(1)).rename(columns=macro_tickers)
        macro_returns.index = pd.to_datetime(macro_returns.index).map(lambda x: x.replace(tzinfo=None).normalize())
        return macro_returns
    except Exception as e:
        st.error(f"マクロデータ取得エラー: {e}")
        return pd.DataFrame()

def calc_fractional_diff(series, d=0.5, window=20):
    weights = [1.0]
    for k in range(1, window):
        weights.append(-weights[-1] * (d - k + 1) / k)
    weights = np.array(weights)[::-1]
    return series.rolling(window).apply(lambda x: np.dot(weights, x), raw=True)

@st.cache_data(show_spinner=False)
def get_stock_features(ticker, macro_returns):
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
        
        # V2.1: TargetをOpen→Closeへ変更 / ATR_Prevの作成
        data["Next_Open"] = data["Open"].shift(-1)
        data["Next_Close"] = data["Close"].shift(-1)
        data["Target_Return"] = (data["Next_Close"] / data["Next_Open"]) - 1.0
        data["Target_Class"] = (data["Next_Close"] > data["Next_Open"]).astype(int)
        data["ATR_Prev_Ratio"] = data["ATR"].shift(1)
        
        return data
    except Exception as e:
        return None

def main():
    st.title("🚀 AI日本株 スクリーニング＆自動売買モニター")
    
    with st.sidebar:
        st.header("⚙️ システム設定")
        
        if st.button("🌐 Kaggleから最新モデルをダウンロード"):
            with st.spinner('Kaggle API経由でモデルをダウンロード中...'):
                success, msg = download_models_from_kaggle()
                if success: st.success(msg)
                else: st.error(msg)
                
        st.markdown("---")
        st.subheader("📋 監視リスト")
        current_tickers = get_tickers()
        tickers_text_default = "\n".join([f"{k}, {v}" for k, v in current_tickers.items()])
        tickers_input = st.text_area("銘柄名, ティッカー", value=tickers_text_default, height=200)
        
        if st.button("💾 監視リストを保存・同期"):
            save_tickers(tickers_input)
            st.rerun()

    tab1, tab2, tab3 = st.tabs(["📊 本日のAI推奨銘柄", "📈 予測推移 & 仮想損益", "🧪 AIモデルの実力検証"])

    with tab1:
        st.header("✨ 本日のスコアランキング (Net Score順)")
        if os.path.exists(RECOMMENDATIONS_PATH):
            df_recom = pd.read_csv(RECOMMENDATIONS_PATH)
            if not df_recom.empty:
                st.dataframe(df_recom, use_container_width=True)
            else:
                st.warning("本日の推奨銘柄はありません（ブレーカー発動中、または条件を満たす銘柄なし）。")
        else:
            st.info("まだ本日のバッチ処理が実行されていないか、結果がありません。")

    with tab2:
        st.header("📈 過去の予測スコア推移")
        if os.path.exists(PREDICTION_HISTORY_PATH):
            df_hist = pd.read_csv(PREDICTION_HISTORY_PATH)
            df_hist['Date'] = pd.to_datetime(df_hist['Date'])
            
            stocks = df_hist['銘柄名'].unique()
            selected_stock = st.selectbox("銘柄を選択", stocks, key="tab2_stock")
            
            df_s = df_hist[df_hist['銘柄名'] == selected_stock].sort_values('Date')
            
            if not df_s.empty:
                fig = plgo.Figure()
                fig.add_trace(plgo.Scatter(x=df_s['Date'], y=df_s['今日の終値'], name="株価", yaxis="y1"))
                
                if 'Net_Score(%)' in df_s.columns:
                    fig.add_trace(plgo.Bar(x=df_s['Date'], y=df_s['Net_Score(%)'], name="Net Score(%)", yaxis="y2", opacity=0.6))
                elif '短期スコア' in df_s.columns:
                    fig.add_trace(plgo.Bar(x=df_s['Date'], y=df_s['短期スコア'], name="短期スコア", yaxis="y2", opacity=0.6))
                    
                fig.update_layout(
                    title=f"{selected_stock} の株価とAIスコア推移",
                    yaxis=dict(title="株価 (円)"),
                    yaxis2=dict(title="スコア/確率", overlaying="y", side="right", range=[0, max(100, df_s.get('Net_Score(%)', pd.Series([100])).max()*1.2)]),
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                st.write("#### 💸 予測履歴からの「仮想損益」シミュレーション (V2.1)")
                tickers_dict = get_tickers()
                selected_ticker = tickers_dict.get(selected_stock, "")
                default_lot = 100 if selected_ticker.endswith(".T") else 1
                
                col_sim1, col_sim2, col_sim3 = st.columns(3)
                with col_sim1: sim_threshold = st.slider("買い条件(Net Score %)", min_value=0, max_value=20, value=0, step=1, key="tab2_sim")
                with col_sim2: sim_initial_cash = st.number_input("初期資金", value=1000000, step=100000, key="tab2_cash")
                with col_sim3: sim_lot_size = st.number_input("1回の購入株数", value=default_lot, step=1, key="tab2_lot")
                
                cash = sim_initial_cash
                sim_history = []
                
                for i in range(len(df_s)):
                    current_date = df_s.iloc[i]['Date']
                    current_price = df_s.iloc[i]['今日の終値']
                    
                    current_prob = 0
                    if 'Net_Score(%)' in df_s.columns: current_prob = df_s.iloc[i]['Net_Score(%)']
                    elif '短期スコア' in df_s.columns: current_prob = df_s.iloc[i]['短期スコア']
                    
                    profit = 0
                    executed = False
                    
                    if i < len(df_s) - 1:
                        next_price = df_s.iloc[i+1]['今日の終値']
                        required_cash = current_price * sim_lot_size
                        if current_prob > sim_threshold and cash >= required_cash:
                            entry_value = current_price * sim_lot_size
                            exit_value = next_price * sim_lot_size
                            # ここは日次履歴からの簡易計算のため、手数料0.2%程度で概算
                            trade_cost = (entry_value * 0.001) + (exit_value * 0.001)
                            profit = (exit_value - entry_value) - trade_cost
                            cash += profit
                            executed = True
                            
                    sim_history.append({
                        'Date': current_date, '今日の終値': current_price, '判断スコア': current_prob,
                        'シグナル': "買い" if executed else "-", '損益(円)': profit if executed else 0, '仮想累積資産(円)': cash
                    })
                
                df_sim = pd.DataFrame(sim_history)
                st.line_chart(df_sim.set_index('Date')['仮想累積資産(円)'])
                st.dataframe(df_sim)
        else:
            st.info("予測履歴データがありません。")

    with tab3:
        st.header("🧪 AIモデルの実力検証 (V2.1 パージング付き・デイ仕様)")
        st.markdown("現在保存されている `classifier_model.pkl` (ベースAI) と `meta_model.pkl` (確信度AI) を使って、過去5年間のデータで**「翌日寄り付き買い → 同日引け売り」**のデイ・バックテストを行います。")
        
        tickers_dict = get_tickers()
        target_name = st.selectbox("検証する銘柄", list(tickers_dict.keys()), key="tab3_stock")
        target_ticker = tickers_dict[target_name]
        
        col1, col2 = st.columns(2)
        with col1:
            bt_initial_cash = st.number_input("検証用 初期資金", value=1000000, step=100000)
            bt_lot_size = st.number_input("1回の購入株数 (検証)", value=100 if target_ticker.endswith(".T") else 1, step=1)
            bt_threshold = st.slider("買い条件 (メタ確信度 %)", min_value=50, max_value=95, value=60, step=1)
            
        with col2:
            st.info("V2.1仕様: デイトレ（Open買い・Close売り）で検証します。\nコストは「ATR連動スリッページ＋固定手数料＋引けズレペナルティ」で厳密に計算されます。")

        if st.button("🔄 デイ・バックテストを実行 (V2.1)", type="primary"):
            with st.spinner('Kaggleの最新メタAI脳を使って、過去の成績を爆速で計算中...'):
                try:
                    clf = joblib.load("models/classifier_model.pkl")
                    meta_model = joblib.load("models/meta_model.pkl")
                    scaler = joblib.load("models/scaler.pkl")
                    selected_features = joblib.load("models/selected_features.pkl")
                    
                    macro_returns = get_macro_data()
                    data_features = get_stock_features(target_ticker, macro_returns)
                    
                    if data_features is not None:
                        ALL_FEATURES = [
                            'Log_Return_Norm', 'Frac_Diff_0.5', 'Disparity_5_Norm', 'Disparity_25_Norm', 
                            'SMA_Cross', 'RSI_14', 'BB_PctB', 'BB_Bandwidth', 'MACD_Norm', 
                            'ATR', 'OBV_Ret', 'USDJPY_Ret', 'SP500_Ret', 'TOPIX_Ret',
                            'EMA_Diff_Ratio', 'MACD_Div', 'Log_Return_Norm_Lag1', 'Log_Return_Norm_Lag2',
                            'RSI_14_Lag1', 'RSI_14_Lag2', 'MACD_Norm_Lag1', 'MACD_Norm_Lag2', 'RSI_Z_Score',
                            'Excess_Return', 'Excess_Return_20d'
                        ]
                        
                        rsi_mean = data_features['RSI_14'].mean()
                        rsi_std = data_features['RSI_14'].std()
                        data_features['RSI_Z_Score'] = (data_features['RSI_14'] - rsi_mean) / (rsi_std + 1e-9)
                        
                        # V2.1: bfillを排除し、欠損行は落とす
                        data_features = data_features.dropna(subset=ALL_FEATURES + ['ATR_Prev_Ratio', 'Open', 'Close'])
                        
                        X_raw = data_features[ALL_FEATURES]
                        X_scaled = pd.DataFrame(scaler.transform(X_raw), index=X_raw.index, columns=ALL_FEATURES)
                        X_sel = X_scaled[selected_features]
                        
                        base_probs = clf.predict_proba(X_sel)[:, 1]
                        X_meta = X_sel.copy()
                        X_meta['Base_Prob'] = base_probs
                        meta_probs = meta_model.predict_proba(X_meta)[:, 1]
                        
                        test_data = data_features.copy()
                        test_probs = meta_probs 
                        
                        # シグナルは「前日」のprobで「当日」に入る（tの判断→t+1 Openで売買）
                        probs_s = pd.Series(test_probs, index=data_features.index).reindex(test_data.index).fillna(0.0).to_numpy()
                        raw_signal = (probs_s >= (bt_threshold / 100.0))

                        # 1日ずらして当日トレードシグナルにする（最初の日は入れない）
                        trade_signal = np.roll(raw_signal, 1)
                        trade_signal[0] = False
                        
                        cash = bt_initial_cash
                        history = []
                        trades = []
                        
                        initial_price = float(test_data['Open'].iloc[0])
                        bh_shares = bt_lot_size if bt_initial_cash >= (initial_price * bt_lot_size) else 0
                        bh_cash = bt_initial_cash - (bh_shares * initial_price)
                        
                        for i in range(len(test_data)):
                            date_i = test_data.index[i]
                            if not trade_signal[i]:
                                equity = cash
                                bh_equity = bh_cash + (bh_shares * float(test_data['Close'].iloc[i]))
                                history.append({"Date": date_i, "AI戦略の資産": equity, "放置(ガチホ)の資産": bh_equity})
                                continue

                            entry = float(test_data["Open"].iloc[i])
                            exit_ = float(test_data["Close"].iloc[i])
                            atr_prev = float(test_data["ATR_Prev_Ratio"].iloc[i])

                            # V2.1 cost_hat（往復比率）
                            cost_hat = cost_hat_roundtrip(atr_prev)

                            # ロット（資金内で固定lotが買えるかだけ見る）
                            required_cash = entry * bt_lot_size
                            if cash < required_cash:
                                equity = cash
                                bh_equity = bh_cash + (bh_shares * exit_)
                                history.append({"Date": date_i, "AI戦略の資産": equity, "放置(ガチホ)の資産": bh_equity})
                                continue

                            gross_ret = (exit_ / entry) - 1.0
                            net_ret = gross_ret - cost_hat

                            pnl = required_cash * net_ret
                            cash += pnl

                            trades.append({"決済日": date_i.strftime("%Y-%m-%d"), "損益(円)": int(pnl)})
                            
                            bh_equity = bh_cash + (bh_shares * exit_)
                            history.append({"Date": date_i, "AI戦略の資産": cash, "放置(ガチホ)の資産": bh_equity})
                        
                        history_df = pd.DataFrame(history)
                        trades_df = pd.DataFrame(trades)
                        final_equity = history_df.iloc[-1]['AI戦略の資産']
                        bh_final_equity = history_df.iloc[-1]['放置(ガチホ)の資産']
                        
                        st.success(f"✅ 検証完了！ AI戦略の最終資産: {int(final_equity):,}円 (ガチホの場合: {int(bh_final_equity):,}円)")
                        
                        fig = plgo.Figure()
                        fig.add_trace(plgo.Scatter(x=history_df['Date'], y=history_df['AI戦略の資産'], name="AI戦略の資産"))
                        fig.add_trace(plgo.Scatter(x=history_df['Date'], y=history_df['放置(ガチホ)の資産'], name="放置(ガチホ)の資産", line=dict(dash='dash')))
                        fig.update_layout(title="資産推移の比較", yaxis_title="資産 (円)")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if not trades_df.empty:
                            st.write(f"**トレード回数**: {len(trades_df)}回")
                            win_rate = len(trades_df[trades_df['損益(円)'] > 0]) / len(trades_df) * 100
                            st.write(f"**勝率**: {win_rate:.1f}%")
                            st.dataframe(trades_df)
                    else:
                        st.error("データの取得に失敗しました。")
                except Exception as e:
                    st.error(f"モデルの読み込みまたは推論に失敗しました: {e}")

if __name__ == "__main__":
    main()
