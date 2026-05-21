import numpy as np
import pandas as pd
from config.simulation_config import SimulationConfig
from src.pricing.black_scholes import BlackScholesPricer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SensitivityAnalyzer:
    """
    Sweeps sigma, strike, and maturity to show how option price
    responds to changes in key parameters.
    """

    def __init__(self, base_config: SimulationConfig):
        self.base = base_config

    def sweep_volatility(self, sigma_range: np.ndarray) -> pd.DataFrame:
        """Price option across a range of volatilities."""
        logger.info(f"Sweeping volatility: {sigma_range[0]:.2f} to {sigma_range[-1]:.2f}")
        records = []
        for sigma in sigma_range:
            cfg = SimulationConfig(
                S0=self.base.S0, K=self.base.K, T=self.base.T,
                r=self.base.r, sigma=sigma,
                n_steps=self.base.n_steps,
                n_simulations=self.base.n_simulations,
                option_type=self.base.option_type
            )
            price = BlackScholesPricer(cfg).price()
            records.append({"sigma": sigma, "price": price})
        return pd.DataFrame(records)

    def sweep_strike(self, strike_range: np.ndarray) -> pd.DataFrame:
        """Price option across a range of strikes."""
        logger.info(f"Sweeping strike: {strike_range[0]:.2f} to {strike_range[-1]:.2f}")
        records = []
        for K in strike_range:
            cfg = SimulationConfig(
                S0=self.base.S0, K=K, T=self.base.T,
                r=self.base.r, sigma=self.base.sigma,
                n_steps=self.base.n_steps,
                n_simulations=self.base.n_simulations,
                option_type=self.base.option_type
            )
            price = BlackScholesPricer(cfg).price()
            records.append({"strike": K, "price": price})
        return pd.DataFrame(records)

    def sweep_maturity(self, maturity_range: np.ndarray) -> pd.DataFrame:
        """Price option across a range of maturities."""
        logger.info(f"Sweeping maturity: {maturity_range[0]:.2f} to {maturity_range[-1]:.2f}")
        records = []
        for T in maturity_range:
            cfg = SimulationConfig(
                S0=self.base.S0, K=self.base.K, T=T,
                r=self.base.r, sigma=self.base.sigma,
                n_steps=max(int(T * 252), 1),
                n_simulations=self.base.n_simulations,
                option_type=self.base.option_type
            )
            price = BlackScholesPricer(cfg).price()
            records.append({"maturity": T, "price": price})
        return pd.DataFrame(records)