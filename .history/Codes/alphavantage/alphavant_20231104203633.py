import numpy as np
import pandas as pd
import requests


def json_to_dataframe_commodities(json_data):
    """
    Converts the given JSON data into a DataFrame with date and value columns for oil prices.
    
    Parameters:
    - json_data (dict): The JSON data with the structure described.
    
    Returns:
    - pd.DataFrame: The DataFrame with date and value columns for oil prices.
    """
    # Extracting data
    data_list = json_data.get('data', [])
    
    # Extracting dates and values
    dates = [entry['date'] for entry in data_list]
    values = [float(entry['value']) for entry in data_list]
    
    # Creating DataFrame
    df = pd.DataFrame({
        'date': dates,
        'value': values
    })

    # Setting date as index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    return df

def json_to_dataframe_price(json_data):
    """
    Converts the given JSON data into a DataFrame with date and OHLC values.
    
    Parameters:
    - json_data (dict): The JSON data with the structure described.
    
    Returns:
    - pd.DataFrame: The DataFrame with date and OHLC values.
    """
    # Dynamically find the key for the time series data
    time_series_key = next((key for key in json_data.keys() if "Time Series" in key), None)
    if time_series_key is None:
        raise ValueError("No 'Time Series' key found in the provided JSON data.")
    
    # Extracting time series data
    time_series_data = json_data.get(time_series_key, {})
    
    # Creating lists to hold data
    dates = []
    open_prices = []
    high_prices = []
    low_prices = []
    close_prices = []

    for date, data in time_series_data.items():
        dates.append(date)
        open_prices.append(float(data['1. open']))
        high_prices.append(float(data['2. high']))
        low_prices.append(float(data['3. low']))
        close_prices.append(float(data['4. close']))

    # Creating DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices
    })

    # Setting date as index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    return df



class AlphaVantage:
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.supported_topics = [
            "blockchain",
            "earnings",
            "ipo",
            "mergers_and_acquisitions",
            "financial_markets",
            "economy_fiscal",
            "economy_monetary",
            "economy_macro",
            "energy_transportation",
            "finance",
            "life_sciences",
            "manufacturing",
            "real_estate",
            "retail_wholesale",
            "technology"
        ]

    def _get_response(self, params, datatype="json"):
        """
        Utility function to send a request and get the response.
        """
        response = requests.get(self.BASE_URL, params=params)
        
        if datatype == "json":
            return response.json()
        else:
            return response.text
        
    def _fetch_data(self, function, symbol, outputsize=None, datatype="json"):
        params = {
            "function": function,
            "symbol": symbol,
            "outputsize": outputsize,
            "datatype": datatype,
            "apikey": self.api_key
        }
        # Filter out None values
        params = {k: v for k, v in params.items() if v is not None}
        response = requests.get(self.BASE_URL, params=params)
        
        if datatype == "json":
            return response.json()
        else:
            return response.text


class FundamentalData(AlphaVantage):
    def __init__(self, api_key):
        super().__init__(api_key)
    
    def get_company_overview(self, symbol, datatype="json"):
        return self._fetch_data("OVERVIEW", symbol, datatype=datatype)

    def get_income_statement(self, symbol, datatype="json"):
        return self._fetch_data("INCOME_STATEMENT", symbol, datatype=datatype)

    def get_balance_sheet(self, symbol, datatype="json"):
        return self._fetch_data("BALANCE_SHEET", symbol, datatype=datatype)

    def get_cash_flow(self, symbol, datatype="json"):
        return self._fetch_data("CASH_FLOW", symbol, datatype=datatype)

    def get_earnings(self, symbol, datatype="json"):
        return self._fetch_data("EARNINGS", symbol, datatype=datatype)

    def get_listing_status(self, date=None, state="active", datatype="csv"):
        params = {
            "function": "LISTING_STATUS",
            "date": date,
            "state": state,
            "apikey": self.api_key,
            "datatype": datatype
        }
        return self._get_response(params, datatype)

    def get_earnings_calendar(self, symbol=None, horizon="3month", datatype="csv"):
        params = {
            "function": "EARNINGS_CALENDAR",
            "symbol": symbol,
            "horizon": horizon,
            "apikey": self.api_key,
            "datatype": datatype
        }
        return self._get_response(params, datatype)

    def get_ipo_calendar(self, datatype="csv"):
        params = {
            "function": "IPO_CALENDAR",
            "apikey": self.api_key,
            "datatype": datatype
        }
        return self._get_response(params, datatype)
    
    
    def get_news_sentiment(self, tickers=None, topics=None, time_from=None, time_to=None, sort="LATEST", limit=50, datatype="json"):
        """
        Fetches the market news & sentiment data based on specified parameters.
        
        :param tickers: The stock/crypto/forex symbols of interest.
        :param topics: The news topics of interest.
        :param time_from: The start time for fetching news articles.
        :param time_to: The end time for fetching news articles.
        :param sort: The sorting order for the articles (default is "LATEST").
        :param limit: The maximum number of articles to fetch (default is 50).
        :param datatype: The desired output format (default is "json").
        :return: The market news & sentiment data.
        """
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": tickers,
            "topics": topics,
            "time_from": time_from,
            "time_to": time_to,
            "sort": sort,
            "limit": limit,
            "apikey": self.api_key,
            "datatype": datatype
        }
        # Filter out None values
        params = {k: v for k, v in params.items() if v is not None}
        response = requests.get(self.BASE_URL, params=params)
        
        if datatype == "json":
            return response.json()
        else:
            return response.text
        


class StocksData(AlphaVantage):
    def __init__(self, api_key):
        super().__init__(api_key)
    
    def get_daily_adjusted_data(self, symbol, outputsize="large", datatype="json"):
        return json_to_dataframe_price(self._fetch_data("TIME_SERIES_DAILY_ADJUSTED", symbol, outputsize, datatype))
    
    def get_weekly_adjusted_data(self, symbol, datatype="json"):
        return json_to_dataframe_price(self._fetch_data("TIME_SERIES_WEEKLY_ADJUSTED", symbol, datatype=datatype))

    def get_monthly_adjusted_data(self, symbol, datatype="json"):
        return json_to_dataframe_price(self._fetch_data("TIME_SERIES_MONTHLY_ADJUSTED", symbol, datatype=datatype))


class CryptocurrencyData(AlphaVantage):
    
    def get_currency_exchange_rate(self, from_currency, to_currency):
        params = {
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": from_currency,
            "to_currency": to_currency,
            "apikey": self.api_key
        }
        return self._get_response(params)

    def get_crypto_intraday(self, symbol, market, interval, outputsize="compact", datatype="json"):
        params = {
            "function": "CRYPTO_INTRADAY",
            "symbol": symbol,
            "market": market,
            "interval": interval,
            "outputsize": outputsize,
            "datatype": datatype,
            "apikey": self.api_key
        }
        return json_to_dataframe_price(self._get_response(params, datatype))

    def get_digital_currency_daily(self, symbol, market):
        params = {
            "function": "DIGITAL_CURRENCY_DAILY",
            "symbol": symbol,
            "market": market,
            "apikey": self.api_key
        }
        return self._get_response(params)

    def get_digital_currency_weekly(self, symbol, market):
        params = {
            "function": "DIGITAL_CURRENCY_WEEKLY",
            "symbol": symbol,
            "market": market,
            "apikey": self.api_key
        }
        return self._get_response(params)

    def get_digital_currency_monthly(self, symbol, market):
        params = {
            "function": "DIGITAL_CURRENCY_MONTHLY",
            "symbol": symbol,
            "market": market,
            "apikey": self.api_key
        }
        return self._get_response(params)



class Commodities(AlphaVantage):
  
    def __init__(self, api_key):
        super().__init__(api_key)

    #By default, interval=monthly. Strings monthly, quarterly, and annual are accepted.
    
    def get_crude_oil_WTI(self, interval="monthly", datatype="json"):
        return json_to_dataframe_commodities(self._fetch_commodity_data("WTI", interval, datatype))

    def get_crude_oil_brent(self, interval="monthly", datatype="json"):
        return json_to_dataframe_commodities(self._fetch_commodity_data("BRENT", interval, datatype))

    def get_natural_gas(self, interval="monthly", datatype="json"):
        return json_to_dataframe_commodities(self._fetch_commodity_data("NATURAL_GAS", interval, datatype))

    def get_copper_price(self, interval="monthly", datatype="json"):
        return json_to_dataframe_commodities(self._fetch_commodity_data("COPPER", interval, datatype))

    def get_aluminum_price(self, interval="monthly", datatype="json"):
        return json_to_dataframe_commodities(self._fetch_commodity_data("ALUMINUM", interval, datatype))

    def get_wheat_price(self, interval="monthly", datatype="json"):
        return json_to_dataframe_commodities(self._fetch_commodity_data("WHEAT", interval, datatype))

    def get_corn_price(self, interval="monthly", datatype="json"):
        return json_to_dataframe_commodities(self._fetch_commodity_data("CORN", interval, datatype))

    def get_cotton_price(self, interval="monthly", datatype="json"):
        return json_to_dataframe_commodities(self._fetch_commodity_data("COTTON", interval, datatype))

    def get_sugar_price(self, interval="monthly", datatype="json"):
        return json_to_dataframe_commodities(self._fetch_commodity_data("SUGAR", interval, datatype))

    def get_coffee_price(self, interval="monthly", datatype="json"):
        return json_to_dataframe_commodities(self._fetch_commodity_data("COFFEE", interval, datatype))

    def get_all_commodities_index(self, interval="monthly", datatype="json"):
        return json_to_dataframe_commodities(self._fetch_commodity_data("ALL_COMMODITIES", interval, datatype))

    def _fetch_commodity_data(self, function, interval, datatype):
        params = {
            "function": function,
            "interval": interval,
            "apikey": self.api_key,
            "datatype": datatype
        }
        response = requests.get(self.BASE_URL, params=params)
        
        if datatype == "json":
            return response.json()
        else:
            return response.text

import requests



class EconomicIndicators(AlphaVantage):

    def __init__(self, api_key):
        super().__init__(api_key)

    def get_real_gdp(self, interval="annual", datatype="json"):
        return json_to_dataframe_commodities(self._fetch_economic_data("REAL_GDP", interval, datatype))

    def get_real_gdp_per_capita(self, datatype="json"):
        return json_to_dataframe_commodities(self._fetch_economic_data("REAL_GDP_PER_CAPITA", datatype=datatype))

    def get_treasury_yield(self, interval="monthly", maturity="10year", datatype="json"):
        return json_to_dataframe_commodities(self._fetch_economic_data("TREASURY_YIELD", interval, datatype, maturity=maturity))

    def get_federal_funds_rate(self, interval="monthly", datatype="json"):
        return json_to_dataframe_commodities(self._fetch_economic_data("FEDERAL_FUNDS_RATE", interval, datatype))

    def get_cpi(self, interval="monthly", datatype="json"):
        return json_to_dataframe_commodities(self._fetch_economic_data("CPI", interval, datatype))

    def get_inflation(self, datatype="json"):
        return json_to_dataframe_commodities(self._fetch_economic_data("INFLATION", datatype=datatype))

    def get_retail_sales(self, datatype="json"):
        return json_to_dataframe_commodities(self._fetch_economic_data("RETAIL_SALES", datatype=datatype))

    def get_durables(self, datatype="json"):
        return json_to_dataframe_commodities(self._fetch_economic_data("DURABLES", datatype=datatype))

    def get_unemployment(self, datatype="json"):
        return json_to_dataframe_commodities(self._fetch_economic_data("UNEMPLOYMENT", datatype=datatype))

    def get_nonfarm_payroll(self, datatype="json"):
        return json_to_dataframe_commodities(self._fetch_economic_data("NONFARM_PAYROLL", datatype=datatype))

    def _fetch_economic_data(self, function, interval=None, datatype="json", maturity=None):
        params = {
            "function": function,
            "interval": interval,
            "apikey": self.api_key,
            "datatype": datatype
        }
        if maturity:
            params["maturity"] = maturity
        
        response = requests.get(self.BASE_URL, params=params)
        
        if datatype == "json":
            return response.json()
        else:
            return response.text




class TechnicalIndicators(AlphaVantage):
    def get_indicator(self, function, symbol, interval, time_period, series_type, month=None, datatype="json", **kwargs):
        params = {
            "function": function,
            "symbol": symbol,
            "interval": interval,
            "time_period": time_period,
            "series_type": series_type,
            "apikey": self.api_key,
            "datatype": datatype
        }
        if month:
            params["month"] = month
        params.update(kwargs)
        
        response = requests.get(self.BASE_URL, params=params)
        return response.json()

    def MAMA(self, symbol, interval, series_type, fastlimit=None, slowlimit=None, month=None, datatype="json"):
        return self.get_indicator("MAMA", symbol, interval, None, series_type, month, datatype, fastlimit=fastlimit, slowlimit=slowlimit)

    def VWAP(self, symbol, interval, month=None, datatype="json"):
        return self.get_indicator("VWAP", symbol, interval, None, None, month, datatype)

    def T3(self, symbol, interval, time_period, series_type, month=None, datatype="json"):
        return self.get_indicator("T3", symbol, interval, time_period, series_type, month, datatype)

    def MACD(self, symbol, interval, series_type, fastperiod=None, slowperiod=None, signalperiod=None, month=None, datatype="json"):
        return self.get_indicator("MACD", symbol, interval, None, series_type, month, datatype, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)

    def MACDEXT(self, symbol, interval, series_type, fastperiod=None, slowperiod=None, signalperiod=None, fastmatype=None, slowmatype=None, signalmatype=None, month=None, datatype="json"):
        return self.get_indicator("MACDEXT", symbol, interval, None, series_type, month, datatype, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod, fastmatype=fastmatype, slowmatype=slowmatype, signalmatype=signalmatype)

    def STOCH(self, symbol, interval, fastkperiod=None, slowkperiod=None, slowdperiod=None, slowkmatype=None, slowdmatype=None, month=None, datatype="json"):
        return self.get_indicator("STOCH", symbol, interval, None, None, month, datatype, fastkperiod=fastkperiod, slowkperiod=slowkperiod, slowdperiod=slowdperiod, slowkmatype=slowkmatype, slowdmatype=slowdmatype)
    def SMA(self, symbol, interval, time_period, series_type, month=None, datatype="json"):
        return self.get_indicator("SMA", symbol, interval, time_period, series_type, month, datatype)
    
    def EMA(self, symbol, interval, time_period, series_type, month=None, datatype="json"):
        return self.get_indicator("EMA", symbol, interval, time_period, series_type, month, datatype)
    
    def WMA(self, symbol, interval, time_period, series_type, month=None, datatype="json"):
        return self.get_indicator("WMA", symbol, interval, time_period, series_type, month, datatype)
    
    def DEMA(self, symbol, interval, time_period, series_type, month=None, datatype="json"):
        return self.get_indicator("DEMA", symbol, interval, time_period, series_type, month, datatype)
    
    def TEMA(self, symbol, interval, time_period, series_type, month=None, datatype="json"):
        return self.get_indicator("TEMA", symbol, interval, time_period, series_type, month, datatype)
    
    def TRIMA(self, symbol, interval, time_period, series_type, month=None, datatype="json"):
        return self.get_indicator("TRIMA", symbol, interval, time_period, series_type, month, datatype)
    
    def KAMA(self, symbol, interval, time_period, series_type, month=None, datatype="json"):
        return self.get_indicator("KAMA", symbol, interval, time_period, series_type, month, datatype)
      
    def STOCHF(self, symbol, interval, fastkperiod=5, fastdperiod=3, fastdmatype=0, month=None, datatype="json"):
        return self.get_indicator("STOCHF", symbol, interval, None, None, month, datatype, fastkperiod=fastkperiod, fastdperiod=fastdperiod, fastdmatype=fastdmatype)

    def RSI(self, symbol, interval, time_period, series_type, month=None, datatype="json"):
        return self.get_indicator("RSI", symbol, interval, time_period, series_type, month, datatype)

    def STOCHRSI(self, symbol, interval, time_period, series_type, fastkperiod=5, fastdperiod=3, fastdmatype=0, month=None, datatype="json"):
        return self.get_indicator("STOCHRSI", symbol, interval, time_period, series_type, month, datatype, fastkperiod=fastkperiod, fastdperiod=fastdperiod, fastdmatype=fastdmatype)

    def WILLR(self, symbol, interval, time_period, month=None, datatype="json"):
        return self.get_indicator("WILLR", symbol, interval, time_period, None, month, datatype)

    def ADX(self, symbol, interval, time_period, month=None, datatype="json"):
        return self.get_indicator("ADX", symbol, interval, time_period, None, month, datatype)

    def ADXR(self, symbol, interval, time_period, month=None, datatype="json"):
            return self.get_indicator("ADXR", symbol, interval, time_period, None, month, datatype)

    def APO(self, symbol, interval, series_type, fastperiod=12, slowperiod=26, matype=0, month=None, datatype="json"):
        return self.get_indicator("APO", symbol, interval, None, series_type, month, datatype, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)

    def PPO(self, symbol, interval, series_type, fastperiod=12, slowperiod=26, matype=0, month=None, datatype="json"):
        return self.get_indicator("PPO", symbol, interval, None, series_type, month, datatype, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)

    def MOM(self, symbol, interval, time_period, series_type, month=None, datatype="json"):
        return self.get_indicator("MOM", symbol, interval, time_period, series_type, month, datatype)
    
    def BOP(self, symbol, interval, month=None, datatype="json"):
        return self.get_indicator("BOP", symbol, interval, None, None, month, datatype)

    def CCI(self, symbol, interval, time_period, month=None, datatype="json"):
        return self.get_indicator("CCI", symbol, interval, time_period, None, month, datatype)

    def CMO(self, symbol, interval, time_period, series_type, month=None, datatype="json"):
        return self.get_indicator("CMO", symbol, interval, time_period, series_type, month, datatype)

    def ROC(self, symbol, interval, time_period, series_type, month=None, datatype="json"):
        return self.get_indicator("ROC", symbol, interval, time_period, series_type, month, datatype)

    def ROCR(self, symbol, interval, time_period, series_type, month=None, datatype="json"):
        return self.get_indicator("ROCR", symbol, interval, time_period, series_type, month, datatype)

    def AROON(self, symbol, interval, time_period, month=None, datatype="json"):
        return self.get_indicator("AROON", symbol, interval, time_period, None, month, datatype)
    
    def AROONOSC(self, symbol, interval, time_period, month=None, datatype="json"):
        return self.get_indicator("AROONOSC", symbol, interval, time_period, None, month, datatype)
    
    def MFI(self, symbol, interval, time_period, month=None, datatype="json"):
        return self.get_indicator("MFI", symbol, interval, time_period, None, month, datatype)

    def TRIX(self, symbol, interval, time_period, series_type, month=None, datatype="json"):
        return self.get_indicator("TRIX", symbol, interval, time_period, series_type, month, datatype)

    def ULTOSC(self, symbol, interval, timeperiod1=7, timeperiod2=14, timeperiod3=28, month=None, datatype="json"):
        # You might need to adjust the get_indicator method or create a specialized one for ULTOSC since it has 3 time periods
        return self.get_indicator("ULTOSC", symbol, interval, timeperiod1, None, month, datatype, timeperiod2, timeperiod3)

    def DX(self, symbol, interval, time_period, month=None, datatype="json"):
        return self.get_indicator("DX", symbol, interval, time_period, None, month, datatype)

    def MINUS_DI(self, symbol, interval, time_period, month=None, datatype="json"):
        return self.get_indicator("MINUS_DI", symbol, interval, time_period, None, month, datatype)
    
    def PLUS_DI(self, symbol, interval, time_period, month=None, datatype="json"):
        return self.get_indicator("PLUS_DI", symbol, interval, time_period, None, month, datatype)
    
    def MINUS_DM(self, symbol, interval, time_period, month=None, datatype="json"):
        return self.get_indicator("MINUS_DM", symbol, interval, time_period, None, month, datatype)

    def PLUS_DM(self, symbol, interval, time_period, month=None, datatype="json"):
        return self.get_indicator("PLUS_DM", symbol, interval, time_period, None, month, datatype)

    def BBANDS(self, symbol, interval, time_period, series_type, nbdevup=2, nbdevdn=2, matype=0, month=None, datatype="json"):
        # Adjust the get_indicator method or create a specialized one for BBANDS due to multiple additional parameters
        return self.get_indicator("BBANDS", symbol, interval, time_period, series_type, month, datatype, nbdevup, nbdevdn, matype)

    def MIDPOINT(self, symbol, interval, time_period, series_type, month=None, datatype="json"):
        return self.get_indicator("MIDPOINT", symbol, interval, time_period, series_type, month, datatype)
    
    
    def MIDPRICE(self, symbol, interval, time_period, month=None, datatype="json"):
        return self.get_indicator("MIDPRICE", symbol, interval, time_period, None, month, datatype)

    def SAR(self, symbol, interval, acceleration=0.01, maximum=0.20, month=None, datatype="json"):
        return self.get_indicator("SAR", symbol, interval, None, None, month, datatype, acceleration, maximum)
    
    def TRANGE(self, symbol, interval, month=None, datatype="json"):
        return self.get_indicator("TRANGE", symbol, interval, None, None, month, datatype)
    
    def ATR(self, symbol, interval, time_period, month=None, datatype="json"):
        return self.get_indicator("ATR", symbol, interval, time_period, None, month, datatype)
    
    def NATR(self, symbol, interval, time_period, month=None, datatype="json"):
        return self.get_indicator("NATR", symbol, interval, time_period, None, month, datatype)
    
    def AD(self, symbol, interval, month=None, datatype="json"):
        return self.get_indicator("AD", symbol, interval, None, None, month, datatype)
    
    def ADOSC(self, symbol, interval, fastperiod=3, slowperiod=10, month=None, datatype="json"):
        return self.get_indicator("ADOSC", symbol, interval, None, None, month, datatype, fastperiod, slowperiod)
    
    def OBV(self, symbol, interval, month=None, datatype="json"):
        return self.get_indicator("OBV", symbol, interval, None, None, month, datatype)
    
    def HT_TRENDLINE(self, symbol, interval, series_type, month=None, datatype="json"):
        return self.get_indicator("HT_TRENDLINE", symbol, interval, None, series_type, month, datatype)
    
    def HT_SINE(self, symbol, interval, series_type, month=None, datatype="json"):
        return self.get_indicator("HT_SINE", symbol, interval, None, series_type, month, datatype)
    
    def HT_TRENDMODE(self, symbol, interval, series_type, month=None, datatype="json"):
        return self.get_indicator("HT_TRENDMODE", symbol, interval, None, series_type, month, datatype)
    
    def HT_DCPERIOD(self, symbol, interval, series_type, month=None, datatype="json"):
        return self.get_indicator("HT_DCPERIOD", symbol, interval, None, series_type, month, datatype)
    
    
    def HT_DCPHASE(self, symbol, interval, series_type, month=None, datatype="json"):
            return self.get_indicator("HT_DCPHASE", symbol, interval, None, series_type, month, datatype)
    
    def HT_PHASOR(self, symbol, interval, series_type, month=None, datatype="json"):
        return self.get_indicator("HT_PHASOR", symbol, interval, None, series_type, month, datatype)