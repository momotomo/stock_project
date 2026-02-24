import yfinance as yf
import matplotlib.pyplot as plt

# トヨタ自動車(7203)の過去1年分のデータを取得
print("データをダウンロードしています...")
ticker = "7203.T"
data = yf.download(ticker, period="1y")

# データの最初の5行を表示して確認
print("ダウンロード完了！データの一部を表示します：")
print(data.head())

# 終値(Close)の折れ線グラフを作成
plt.figure(figsize=(10, 5))
# yfinanceの新しいバージョンに対応するため、Closeの列だけを指定
plt.plot(data.index, data['Close'], label='Toyota (7203.T)')
plt.title('Toyota Motor Corp - 1 Year Stock Price')
plt.xlabel('Date')
plt.ylabel('Price (JPY)')
plt.grid(True)
plt.legend()
plt.show()