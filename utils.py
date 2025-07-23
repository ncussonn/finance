import yfinance as yf
import numpy as np
import fear_and_greed
import datetime
from dateutil.relativedelta import relativedelta # used to prevent leap year from causing issues when looking back in time on Feb. 29th
import matplotlib.pyplot as plt

def create_start_end_dates(time_range: list):
    if time_range is None:
        # Use 1Y if no range provided
        end_date = datetime.date.today()
        start_date = end_date - relativedelta(years=1)
    else:
        start_date, end_date = time_range[0], time_range[1]

    return start_date, end_date

def get_max_min(yf_data, current_value: float):
    highest_values = yf_data['High']
    lowest_values = yf_data['Low']
    high = np.max(highest_values)
    low = np.min(lowest_values)

    if current_value > high:
        high = current_value
    if current_value < low:
        low = current_value

    return high, low

def snp500(time_range: list = None, print_statements: bool = True, plot: bool = True):

    start_date, end_date = create_start_end_dates(time_range)

    snp500_data = yf.download("^GSPC", start=start_date, end=end_date, progress=False, auto_adjust=False)
    snp500_close = snp500_data['Close']
    snp500_live = yf.Ticker("^GSPC")
    current_snp500 = snp500_live.info['regularMarketPrice']

    if print_statements:
        print(f"Current S&P 500 Index: {current_snp500:.2f}")

    if plot:
        ma_20 = snp500_close.rolling(window=20, min_periods=1).mean()
        ma_50 = snp500_close.rolling(window=50, min_periods=1).mean()
        ma_alpha = 0.5

        plt.figure()
        plt.plot(snp500_data.index, snp500_close, label = "Index")
        plt.plot(snp500_data.index, ma_20, label = "20-Day MA", alpha = ma_alpha)
        plt.plot(snp500_data.index, ma_50, label = "50-Day MA", alpha = ma_alpha)
        plt.grid(True)
        plt.xlabel("Date")
        plt.ylabel("Index Value")
        plt.title("S&P 500 Index")
        plt.legend()
        plt.show()

    return snp500_data

def bonds(time_range: list = None, print_statements: bool = True, plot: bool = True):

    start_date, end_date = create_start_end_dates(time_range)

    bnd_data = yf.download("BND", start=start_date, end=end_date, progress=False, auto_adjust=False)
    shy_data = yf.download("SHY", start=start_date, end=end_date, progress=False, auto_adjust=False)
    iei_data = yf.download("IEI", start=start_date, end=end_date, progress=False, auto_adjust=False)
    ief_data = yf.download("IEF", start=start_date, end=end_date, progress=False, auto_adjust=False)
    tlh_data = yf.download("TLH", start=start_date, end=end_date, progress=False, auto_adjust=False)
    tlt_data = yf.download("TLT", start=start_date, end=end_date, progress=False, auto_adjust=False)

    if plot:
        non_aggregate_alpha = 0.5
        plt.figure()
        plt.title("Bonds")
        plt.plot(bnd_data.index, bnd_data['Close']/bnd_data['Close'].iloc[0], label = "Aggregate")
        plt.plot(shy_data.index, shy_data['Close']/shy_data['Close'].iloc[0], label = '1-3 Year', alpha = non_aggregate_alpha)
        plt.plot(iei_data.index, iei_data['Close']/iei_data['Close'].iloc[0], label = '3-7 Year', alpha = non_aggregate_alpha)
        plt.plot(ief_data.index, ief_data['Close']/ief_data['Close'].iloc[0], label = '7-10 Year', alpha = non_aggregate_alpha)
        plt.plot(tlh_data.index, tlh_data['Close']/tlh_data['Close'].iloc[0], label = '10-20 Year', alpha = non_aggregate_alpha)
        plt.plot(tlt_data.index, tlt_data['Close']/tlt_data['Close'].iloc[0], label = '20+ Year', alpha = non_aggregate_alpha)
        plt.grid(True)
        plt.xlabel("Date")
        plt.ylabel("Normalized Value")
        plt.legend()
        plt.show()

    return bnd_data

def gold_silver(time_range: list = None, print_statements: bool = True, plot: bool = True):

    start_date, end_date = create_start_end_dates(time_range)

    gold_data = yf.download("GC=F", start=start_date, end=end_date, progress=False, auto_adjust=False)
    silv_data = yf.download("SI=F", start=start_date, end=end_date, progress=False, auto_adjust=False)

    current_gold = yf.Ticker("GC=F").info['regularMarketPrice']
    gold_high, gold_low = get_max_min(gold_data, current_gold)
    current_silv = yf.Ticker("SI=F").info['regularMarketPrice']
    silver_high, silver_low = get_max_min(silv_data, current_silv)
    print("Gold\n_____")
    print(f"Highest Value: {gold_high:.2f}")
    print(f"Lowest Value: {gold_low:.2f}")
    print("Silver\n_____")
    print(f"Highest Value: {silver_high:.2f}")
    print(f"Lowest Value: {silver_low:.2f}")
    
    if plot:
        fig, axs = plt.subplots(2,1, sharex=True)
        axs[0].plot(gold_data.index, gold_data['Close'])
        axs[1].plot(silv_data.index, silv_data['Close'])
        axs[0].grid(True)
        axs[1].grid(True)
        axs[0].set_title('Gold and Silver Prices')
        axs[1].set_xlabel('Date')
        axs[0].set_ylabel('Gold USD/oz')
        axs[1].set_ylabel('Silver USD/oz')
        plt.show()

    return gold_data, silv_data


def gold_silv_ratio(print_statements: bool = True):
    '''Gets the current gold to silver ratio.'''
    gold = yf.Ticker("GC=F")    # Gold Futures
    silver = yf.Ticker("SI=F")  # Silver Futures

    gold_price = gold.info.get('regularMarketPrice')
    silver_price = silver.info.get('regularMarketPrice')
    gold_to_silv_ratio = gold_price / silver_price

    if print_statements:
        print(f"Gold: ${gold_price:.2f} per ounce")
        print(f"Silver: ${silver_price:.2f} per ounce")
        print(f"Gold/Silver Ratio: {gold_to_silv_ratio:.2f}")
    
        if gold_to_silv_ratio > 80:
            print(f"Silver likely undervalued.")
        elif gold_to_silv_ratio < 60:
            print(f"Gold likely undervalued.")
        else:
            print(f"Unclear if gold or silver undervalued.")

    return gold_to_silv_ratio

def fear_n_greed_idx(print_statements: bool = True):
    '''Gets fear and greed index data from CNN'''
    index_data = fear_and_greed.get()
    if print_statements:
        print(f"Fear & Greed Index Value: {index_data.value:.2f}")
        print(f"Market Sentiment: {index_data.description}")

    return index_data

def vix(time_range: list = None, print_statements: bool = True, plot: bool = True):
    '''Gets the CBOE Volatilty Index'''
    start_date, end_date = create_start_end_dates(time_range)

    # Download VIX data
    vix_data = yf.download("^VIX", start=start_date, end=end_date, progress=False, auto_adjust=False)

    # Access specific columns, e.g., 'Close' price of VIX
    vix_close_prices = vix_data['Close']
    
    # Get the latest intraday VIX value (real-time or slightly delayed)
    vix_live = yf.Ticker("^VIX")
    current_vix = vix_live.info['regularMarketPrice']
    high_volatility_value = 30
    stable_volatility_value = 20

    highest, lowest = get_max_min(vix_data, current_vix)

    if print_statements:
        # Print the first few rows of the data
        print(f"Current VIX: {current_vix:.2f}")
        print(f"Lowest Value: {lowest:.2f}")
        print(f"Highest Value: {highest:.2f}")
        if current_vix >= high_volatility_value:
            print("High fear / volatility market.")
        if current_vix <= stable_volatility_value:
            print("Stable market conditions.")
        if current_vix < high_volatility_value and current_vix > stable_volatility_value:
            print("Cautious market conditions. Neither stable nor fearful.")

    if plot:
        plt.figure()
        plt.title("CBOE Volatility Index")
        plt.plot(vix_data.index, vix_close_prices, label = 'VIX')
        plt.axhline(stable_volatility_value, color = 'green', linestyle='-.', label = "stable")
        plt.axhline(high_volatility_value, color = 'red', linestyle='-.', label = "fearful / volatile")
        plt.xlabel('Date')
        plt.ylabel('Index Value')
        plt.legend()
        plt.grid(True)
        plt.show()

    return vix_data

def bitcoin(time_range: list = None, print_statements: bool = True, plot: bool = True):
    '''Historic Price of Bitcoin'''
    start_date, end_date = create_start_end_dates(time_range)

    btc_data = yf.download("BTC-USD", start=start_date, end=end_date, progress=False, auto_adjust=False)
    btc_data = btc_data.dropna()

    # Latest BTC Value
    btc_live = yf.Ticker("BTC-USD")
    current_btc = btc_live.info['regularMarketPrice']

    highest, lowest = get_max_min(btc_data, current_btc)

    if print_statements:
        print(f"Current Bitcoin Price (USD): {current_btc:.2f}")
        print(f"Lowest Price: {lowest:.2f}")
        print(f"Highest Price: {highest:.2f}")

    if plot:
        plt.figure()
        plt.title("Bitcoin in USD")
        plt.plot(btc_data.index, btc_data["Close"])
        plt.xlabel('Date')
        plt.ylabel('USD')
        plt.grid(True)
        plt.show()

    return btc_data