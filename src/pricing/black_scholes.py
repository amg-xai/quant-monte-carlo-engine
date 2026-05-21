import numpy as np
from scipy.stats import norm
from config.simulation_config import SimulationConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BlackScholesPricer:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self._d1, self._d2 = self._compute_d1_d2()

    def _compute_d1_d2(self) -> tuple:
        cfg = self.config
        d1 = (
            np.log(cfg.S0 / cfg.K) +
            (cfg.r + 0.5 * cfg.sigma ** 2) * cfg.T
        ) / (cfg.sigma * np.sqrt(cfg.T))
        d2 = d1 - cfg.sigma * np.sqrt(cfg.T)
        return d1, d2

    def price(self) -> float:
        cfg = self.config
        d1, d2 = self._d1, self._d2
        if cfg.option_type == "call":
            price = (cfg.S0 * norm.cdf(d1) - cfg.K * np.exp(-cfg.r * cfg.T) * norm.cdf(d2))
        elif cfg.option_type == "put":
            price = (cfg.K * np.exp(-cfg.r * cfg.T) * norm.cdf(-d2) - cfg.S0 * norm.cdf(-d1))
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got '{cfg.option_type}'")
        logger.info(f"Black-Scholes {cfg.option_type} price: {price:.6f}")
        return price

    def delta(self) -> float:
        if self.config.option_type == "call":
            return norm.cdf(self._d1)
        return norm.cdf(self._d1) - 1

    def gamma(self) -> float:
        cfg = self.config
        return norm.pdf(self._d1) / (cfg.S0 * cfg.sigma * np.sqrt(cfg.T))

    def vega(self) -> float:
        cfg = self.config
        return cfg.S0 * norm.pdf(self._d1) * np.sqrt(cfg.T) * 0.01

    def theta(self) -> float:
        cfg = self.config
        d1, d2 = self._d1, self._d2
        term1 = -(cfg.S0 * norm.pdf(d1) * cfg.sigma) / (2 * np.sqrt(cfg.T))
        if cfg.option_type == "call":
            term2 = cfg.r * cfg.K * np.exp(-cfg.r * cfg.T) * norm.cdf(d2)
            return (term1 - term2) / 365
        else:
            term2 = cfg.r * cfg.K * np.exp(-cfg.r * cfg.T) * norm.cdf(-d2)
            return (term1 + term2) / 365

    def summary(self) -> dict:
        return {
            "option_type": self.config.option_type,
            "price": self.price(),
            "delta": self.delta(),
            "gamma": self.gamma(),
            "vega": self.vega(),
            "theta": self.theta(),
        }