import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from src.utils.logger import get_logger

logger = get_logger(__name__)


def black_scholes_price(S, K, T, r, sigma, option_type="call") -> float:
    """Compute BS price for a given sigma."""
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_volatility(market_price: float, S: float, K: float,
                       T: float, r: float, option_type: str = "call") -> float:
    """
    Extract implied volatility by inverting the Black-Scholes formula.
    Uses Brent's method to find sigma such that BS(sigma) = market_price.
    Returns NaN if no solution found.
    """
    if T <= 0 or market_price <= 0:
        return np.nan

    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    if market_price < intrinsic * 0.999:
        return np.nan

    try:
        iv = brentq(
            lambda sigma: black_scholes_price(S, K, T, r, sigma, option_type) - market_price,
            a=1e-6,
            b=10.0,
            xtol=1e-6,
            maxiter=500
        )
        return iv
    except (ValueError, RuntimeError):
        return np.nan


def compute_iv_surface(chain, S: float, r: float, T: float) -> "pd.DataFrame":
    """
    Compute implied volatility for every contract in the options chain.
    """
    import pandas as pd

    logger.info("Computing implied volatilities...")

    ivs = []
    for _, row in chain.iterrows():
        iv = implied_volatility(
            market_price=row["market_price"],
            S=S,
            K=row["strike"],
            T=T,
            r=r,
            option_type=row["option_type"]
        )
        ivs.append(iv)

    chain = chain.copy()
    chain["computed_iv"] = ivs

    valid = chain["computed_iv"].notna().sum()
    logger.info(f"IV computed for {valid}/{len(chain)} contracts")

    return chain