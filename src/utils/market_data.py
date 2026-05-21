import yfinance as yf
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MarketDataFetcher:
    """
    Fetches live stock prices and options chains from Yahoo Finance.
    Computes historical volatility from price returns.
    """

    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)

    def get_spot_price(self) -> float:
        info = self.stock.fast_info
        price = info.last_price
        logger.info(f"{self.ticker} spot price: ${price:.2f}")
        return float(price)

    def get_historical_volatility(self, period: str = "1y") -> float:
        hist = self.stock.history(period=period)
        log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        daily_vol = log_returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        logger.info(f"{self.ticker} historical volatility ({period}): {annual_vol:.4f}")
        return float(annual_vol)

    def get_options_chain(self, expiry: str = None) -> pd.DataFrame:
        expiries = self.stock.options

        if not expiries:
            raise ValueError(f"No options data available for {self.ticker}")

        if expiry is None:
            expiry = expiries[0]
            logger.info(f"No expiry specified. Using nearest: {expiry}")

        logger.info(f"Fetching options chain for {self.ticker} expiry {expiry}...")

        chain = self.stock.option_chain(expiry)

        calls = chain.calls.copy()
        calls["option_type"] = "call"

        puts = chain.puts.copy()
        puts["option_type"] = "put"

        combined = pd.concat([calls, puts], ignore_index=True)

        cols = ["strike", "lastPrice", "bid", "ask", "impliedVolatility",
                "volume", "openInterest", "option_type"]
        combined = combined[cols].copy()
        combined.rename(columns={"lastPrice": "market_price",
                                  "impliedVolatility": "market_iv"}, inplace=True)

        combined = combined[combined["volume"] > 0].reset_index(drop=True)

        logger.info(f"Options chain fetched: {len(combined)} contracts")
        return combined

    def get_available_expiries(self) -> list:
        expiries = list(self.stock.options)
        logger.info(f"{self.ticker} available expiries: {expiries}")
        return expiries