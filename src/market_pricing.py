import numpy as np
from datetime import datetime
from config.simulation_config import SimulationConfig
from src.utils.market_data import MarketDataFetcher
from src.analytics.implied_volatility import compute_iv_surface
from src.pricing.black_scholes import BlackScholesPricer
from src.pricing.monte_carlo_pricer import MonteCarloPricer
from src.visualization.plotter import Plotter
from src.utils.logger import get_logger

logger = get_logger(__name__)

RISK_FREE_RATE = 0.05


def run_market_pricing(ticker: str, option_type: str = "call"):

    logger.info("=" * 55)
    logger.info(f"  MARKET PRICING — {ticker}")
    logger.info("=" * 55)

    fetcher = MarketDataFetcher(ticker)
    S0      = fetcher.get_spot_price()
    sigma   = fetcher.get_historical_volatility(period="1y")

    expiries = fetcher.get_available_expiries()
    expiry   = expiries[0]

    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
    T = max((expiry_dt - datetime.today()).days / 365, 1/365)

    logger.info(f"Expiry: {expiry} | T = {T:.4f} years")

    chain = fetcher.get_options_chain(expiry=expiry)
    chain = compute_iv_surface(chain, S=S0, r=RISK_FREE_RATE, T=T)

    calls = chain[chain["option_type"] == option_type].dropna(subset=["computed_iv"])
    atm_row = calls.iloc[(calls["strike"] - S0).abs().argsort()[:1]]

    K         = float(atm_row["strike"].values[0])
    market_px = float(atm_row["market_price"].values[0])
    market_iv = float(atm_row["computed_iv"].values[0])

    logger.info("-" * 55)
    logger.info(f"ATM Strike selected : ${K:.2f}")
    logger.info(f"Market price        : ${market_px:.4f}")
    logger.info(f"Market implied vol  : {market_iv*100:.2f}%")
    logger.info(f"Historical vol      : {sigma*100:.2f}%")
    logger.info("-" * 55)

    config_iv = SimulationConfig(
        S0=S0, K=K, T=T, r=RISK_FREE_RATE,
        sigma=market_iv,
        n_steps=max(int(T * 252), 1),
        n_simulations=100_000,
        option_type=option_type
    )

    bs_iv       = BlackScholesPricer(config_iv)
    mc_iv       = MonteCarloPricer(config_iv)
    bs_price_iv = bs_iv.price()
    mc_result   = mc_iv.price(random_seed=42)

    logger.info("PRICING WITH IMPLIED VOL")
    logger.info(f"  Market price    : ${market_px:.4f}")
    logger.info(f"  BS  price (IV)  : ${bs_price_iv:.4f}")
    logger.info(f"  MC  price (IV)  : ${mc_result['price']:.4f}")
    logger.info(f"  MC  95% CI      : [${mc_result['ci_lower']:.4f}, ${mc_result['ci_upper']:.4f}]")
    logger.info(f"  BS vs Market    : ${abs(bs_price_iv - market_px):.4f} diff")
    logger.info(f"  MC vs Market    : ${abs(mc_result['price'] - market_px):.4f} diff")
    logger.info("-" * 55)

    config_hv = SimulationConfig(
        S0=S0, K=K, T=T, r=RISK_FREE_RATE,
        sigma=sigma,
        n_steps=max(int(T * 252), 1),
        n_simulations=100_000,
        option_type=option_type
    )

    bs_hv       = BlackScholesPricer(config_hv)
    bs_price_hv = bs_hv.price()

    logger.info("PRICING WITH HISTORICAL VOL")
    logger.info(f"  BS price (HV)   : ${bs_price_hv:.4f}")
    logger.info(f"  HV vs Market    : ${abs(bs_price_hv - market_px):.4f} diff")
    logger.info(f"  IV vs HV diff   : {abs(market_iv - sigma)*100:.2f}% vol spread")
    logger.info("-" * 55)

    # --- Volatility smile plot ---
    plotter = Plotter(config_iv)
    plotter.plot_volatility_smile(
        chain=chain,
        spot_price=S0,
        title=f"{ticker} | Expiry {expiry}"
    )

    # --- Volatility surface across all expiries ---
    logger.info("Building volatility surface across all expiries...")
    surface_data = []

    for exp in expiries[:6]:  # limit to 6 expiries to keep it fast
        try:
            exp_dt = datetime.strptime(exp, "%Y-%m-%d")
            T_exp  = max((exp_dt - datetime.today()).days / 365, 1/365)
            ch     = fetcher.get_options_chain(expiry=exp)
            ch     = compute_iv_surface(ch, S=S0, r=RISK_FREE_RATE, T=T_exp)

            for _, row in ch.iterrows():
                if row["computed_iv"] > 0 and row["computed_iv"] < 5:
                    surface_data.append({
                        "expiry":      exp,
                        "strike":      row["strike"],
                        "iv":          row["computed_iv"],
                        "option_type": row["option_type"]
                    })
        except Exception as e:
            logger.warning(f"Skipping expiry {exp}: {e}")
            continue

    if surface_data:
        plotter.plot_volatility_surface(
            surface_data=surface_data,
            spot_price=S0,
            ticker=ticker
        )

    logger.info("=" * 55)
    logger.info("Market pricing complete.")
    logger.info("=" * 55)


if __name__ == "__main__":
    import sys
    ticker      = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    option_type = sys.argv[2] if len(sys.argv) > 2 else "call"
    run_market_pricing(ticker, option_type)