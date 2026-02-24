import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit

# --- トリプルバリア法のラベリング関数 ---
@st.cache_data
def calc_triple_barrier(prices, horizon, pt_pct, sl_pct):
    """
    指定期間(horizon)内に、上部バリア(pt_pct: 利確)または下部バリア(sl_pct: 損切)に到達するかを判定。
    1: 上部に先回到達 (利確)
    -1: 下部に先回到達 (損切り)
    0: どちらにも到達せず時間切れ
    """
    vals = prices.values
    n = len(vals)
    labels = np.full(n, np.nan)
    
    for i in range(n - horizon):
        p0 = vals[i]
        ub = p0 * (1 + pt_pct) # 利確ライン
        lb = p0 * (1 - sl_pct) # 損切りライン
        
        path = vals[i+1 : i+1+horizon]
        
        hit_ub = np.where(path >= ub)[0]
        hit_lb = np.where(path <= lb)[0]
        
        first_ub = hit_ub[0] if len(hit_ub) > 0 else horizon + 1
        first_lb = hit_lb[0] if len(hit_lb) > 0 else horizon + 1
        
        if first_ub == (horizon + 1) and first_lb == (horizon + 1):
            labels[i] = 0  # 時間切れ
        elif first_ub < first_lb:
            labels[i] = 1  # 利確
        elif first_lb < first_ub:
            labels[i] = -1 # 損切
        else:
            labels[i] = 0  # 同時到達（安全のため0）
            
    return labels

# ページ全体を広く使う設定
st.set_page_config(layout="wide")

st.title("🚀 AI株価スクリーニングダッシュボード (中長期トリプルバリア版)")
st.write("短期の予測に加え、中長期スパンでの「損切り前に利確ラインに到達する確率」を複数のAIモデルで並行予測します。")

# 🌟 特徴量と評価手法の解説
with st.expander("🧠 AIが学習に使用している特徴量と「トリプルバリア法」について"):
    st.markdown("""
    ### 📊 使用している特徴量（絶対価格を排除した相対指標）
    1. **Log_Return**: 対数収益率　2. **Disparity_5/25**: SMA乖離率　3. **SMA_Cross**: 移動平均クロス
    4. **RSI_14**: オシレーター　5. **BB_PctB / Bandwidth**: ボリンジャーバンド情報
    6. **MACD_Norm**: 正規化MACD　7. **ATR**: ボラティリティ　8. **OBV_Ret**: 出来高フロー
    9. **USDJPY_Ret / SP500_Ret**: ドル円・S&P500の変動（マクロ指標）
    
    ---
    ### 🎯 トリプルバリア法 (Triple Barrier Method) とは？
    中長期の投資判断をAIに学習させるための、金融データサイエンス特有のラベリング手法です。
    特定の期間（時間バリア）内に、設定した**「利確ライン（上部バリア）」**に先に到達すれば「成功(1)」、**「損切りライン（下部バリア）」**に先に到達すれば「失敗(-1)」、どちらにも触れずに期間を終えれば「時間切れ(0)」としてAIに過去のチャートの答えを教え込みます。
    
    本システムでは、期間に応じて以下のバリア幅を自動設定して予測を行っています。
    * **1W (1週間 / 5営業日)**: 利確 +3% / 損切 -3%
    * **2W (2週間 / 10営業日)**: 利確 +5% / 損切 -5%
    * **1M (1ヶ月 / 21営業日)**: 利確 +10% / 損切 -10%
    * **3M (3ヶ月 / 63営業日)**: 利確 +20% / 損切 -20%
    * **6M (半年 / 126営業日)**: 利確 +30% / 損切 -30%
    * **1Y (1年 / 252営業日)**: 利確 +50% / 損切 -50%
    """)

# 🌟 新機能（フェーズ1）：サイドバーで分析対象の銘柄を自由に編集できるUIを追加
st.sidebar.header("⚙️ 分析銘柄の設定")
st.sidebar.write("分析したい銘柄を「銘柄名, ティッカー」の形式で追加・変更できます。")

default_tickers_str = """トヨタ自動車, 7203.T
三菱UFJ, 8306.T
ソニーG, 6758.T
ソフトバンクG, 9984.T
任天堂, 7974.T"""

ticker_input = st.sidebar.text_area(
    "監視リスト", 
    value=default_tickers_str, 
    height=250,
    help="カンマ区切りで銘柄名とYahoo Financeのティッカーを入力してください。日本株は証券コードの末尾に「.T」を付けます。"
)

# 入力されたテキストを辞書に変換
tickers = {}
for line in ticker_input.strip().split('\n'):
    line = line.strip()
    if ',' in line:
        name, ticker = line.split(',', 1)
        tickers[name.strip()] = ticker.strip()

if not tickers:
    st.warning("👈 左側のサイドバーから分析する銘柄を少なくとも1つ入力してください。")

if st.button("🔍 プロ仕様の分析を実行する", type="primary") and tickers:
    with st.spinner('マクロデータ取得中... および、7つの期間（明日・1W・2W・1M・3M・6M・1Y）のAIモデルを並行学習しています...（データ量が多いため少し時間がかかります）'):
        results = []
        
        # --- 🌐 マクロ経済データの取得 (5年分) ---
        macro_tickers = {"USDJPY=X": "USDJPY_Ret", "^GSPC": "SP500_Ret"}
        macro_data = yf.download(list(macro_tickers.keys()), period="5y", progress=False)['Close']
        if isinstance(macro_data.columns, pd.MultiIndex):
            macro_data.columns = macro_data.columns.get_level_values(0)
        macro_returns = np.log(macro_data / macro_data.shift(1)).rename(columns=macro_tickers)
        
        for name, ticker in tickers.items():
            stock_info = yf.Ticker(ticker).info
            per = stock_info.get('trailingPE', None) 
            
            # --- 個別株データの取得 (5年分) ---
            data = yf.download(ticker, period="5y", progress=False)
            if data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
                
            # --- 🛠️ 高度な特徴量エンジニアリング ---
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
            data = data.join(macro_returns)
            
            # --- 🎯 従来のターゲット（明日の予測用） ---
            data['Target_Class'] = (data['Close'].shift(-1) > data['Close']).astype(int)
            data['Target_Price'] = data['Close'].shift(-1)
            
            features = ['Log_Return', 'Disparity_5', 'Disparity_25', 'SMA_Cross', 'RSI_14', 
                        'BB_PctB', 'BB_Bandwidth', 'MACD_Norm', 'ATR', 'OBV_Ret', 'USDJPY_Ret', 'SP500_Ret']
            
            # 従来モデル用のクリーンデータ
            data_clean_short = data.dropna(subset=features + ['Target_Class', 'Target_Price'])
            
            if len(data_clean_short) > 100:
                # ---------------------------------------------
                # モデル1＆2: 従来通り「明日」の分類と回帰
                # ---------------------------------------------
                historical_data = data_clean_short[:-1]
                latest_data = data_clean_short.iloc[[-1]]
                current_price = latest_data['Close'].values[0]
                
                X_raw = historical_data[features]
                y_class = historical_data['Target_Class']
                y_price = historical_data['Target_Price']
                
                scaler = RobustScaler()
                X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), index=X_raw.index, columns=features)
                latest_X_scaled = pd.DataFrame(scaler.transform(latest_data[features]), index=latest_data.index, columns=features)
                
                # WF検証（明日の予測の信頼度）
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
                expected_change_pct = ((pred_price - current_price) / current_price) * 100
                
                # ---------------------------------------------
                # モデル3〜8: トリプルバリア法（中長期予測）
                # ---------------------------------------------
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
                    
                    # そのスパンの未来が「確定」している過去のデータだけを抽出して学習
                    valid_idx = data[data[target_col].notna()].index
                    train_idx = valid_idx.intersection(X_raw.index)
                    
                    if len(train_idx) > 100:
                        X_train_h = X_scaled.loc[train_idx]
                        y_train_h = data.loc[train_idx, target_col]
                        
                        clf_h = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                        clf_h.fit(X_train_h, y_train_h)
                        
                        classes = list(clf_h.classes_)
                        # 「1 (利確バリアに到達)」の確率を取得
                        if 1 in classes:
                            idx_1 = classes.index(1)
                            prob_1 = clf_h.predict_proba(latest_X_scaled)[0][idx_1]
                        else:
                            prob_1 = 0.0
                    else:
                        prob_1 = 0.0
                        
                    tb_results[h_params['name']] = float(prob_1 * 100)
                
                # ---------------------------------------------
                # 結果の格納
                # ---------------------------------------------
                per_str = f"{round(per, 1)}倍" if isinstance(per, (int, float)) else "-"
                result_dict = {
                    "銘柄名": name,
                    "WF平均正解率(明日)": float(wf_mean_accuracy * 100),
                    "PER(割安)": per_str,
                    "今日の終値": float(current_price),
                    "明日の予測値": float(pred_price),
                    "明日の上昇確率": float(prob_up * 100)
                }
                # 中長期のトリプルバリア結果を合体
                result_dict.update(tb_results)
                results.append(result_dict)
        
        # --- ⑤ 結果の表示 ---
        if results:
            df_results = pd.DataFrame(results)
            # 全体的に「1ヶ月後に10%上がる確率」が高いものをトップとみなしてソート
            df_results = df_results.sort_values(by="1M 利確(>10%)", ascending=False).reset_index(drop=True)
            
            st.divider()
            
            st.write("### 🏆 本日のAIイチオシ銘柄（1ヶ月後の中期見通しトップ）")
            top_stock = df_results.iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("銘柄名", top_stock['銘柄名'])
            col2.metric("明日の予測値", f"¥{top_stock['明日の予測値']:,.1f}")
            col3.metric("明日の上昇確率", f"{top_stock['明日の上昇確率']:.1f}%")
            col4.metric("1ヶ月後の利確(>10%)確率", f"{top_stock['1M 利確(>10%)']:.1f}%")
            
            st.write("<br>", unsafe_allow_html=True)
            
            st.write("### 📊 全銘柄のスクリーニング結果詳細（短期〜中長期）")
            st.dataframe(
                df_results,
                column_config={
                    "銘柄名": st.column_config.TextColumn("銘柄名"),
                    "WF平均正解率(明日)": st.column_config.ProgressColumn(
                        "WF正解率(明日)", min_value=0, max_value=100, format="%.1f%%"
                    ),
                    "PER(割安)": st.column_config.TextColumn("PER"),
                    "今日の終値": st.column_config.NumberColumn("終値", format="¥ %.1f"),
                    "明日の予測値": st.column_config.NumberColumn("明日の予測値", format="¥ %.1f"),
                    "明日の上昇確率": st.column_config.ProgressColumn(
                        "明日 上がる確率", min_value=0, max_value=100, format="%.1f%%"
                    ),
                    "1W 利確(>3%)": st.column_config.ProgressColumn(
                        "1週間後 利確(>3%)", help="1週間以内に損切(-3%)より先に利確(+3%)に達する確率", min_value=0, max_value=100, format="%.1f%%"
                    ),
                    "2W 利確(>5%)": st.column_config.ProgressColumn(
                        "2週間後 利確(>5%)", help="2週間以内に損切(-5%)より先に利確(+5%)に達する確率", min_value=0, max_value=100, format="%.1f%%"
                    ),
                    "1M 利確(>10%)": st.column_config.ProgressColumn(
                        "1ヶ月後 利確(>10%)", help="1ヶ月以内に損切(-10%)より先に利確(+10%)に達する確率", min_value=0, max_value=100, format="%.1f%%"
                    ),
                    "3M 利確(>20%)": st.column_config.ProgressColumn(
                        "3ヶ月後 利確(>20%)", help="3ヶ月以内に損切(-20%)より先に利確(+20%)に達する確率", min_value=0, max_value=100, format="%.1f%%"
                    ),
                    "6M 利確(>30%)": st.column_config.ProgressColumn(
                        "半年後 利確(>30%)", help="半年以内に損切(-30%)より先に利確(+30%)に達する確率", min_value=0, max_value=100, format="%.1f%%"
                    ),
                    "1Y 利確(>50%)": st.column_config.ProgressColumn(
                        "1年後 利確(>50%)", help="1年以内に損切(-50%)より先に利確(+50%)に達する確率", min_value=0, max_value=100, format="%.1f%%"
                    ),
                },
                hide_index=True, 
                use_container_width=True
            )
            
            st.caption("※表の項目名をクリックすると、並び替えができます。横スクロールで中長期の確率まで確認できます。")