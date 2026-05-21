import numpy as np
from config.simulation_config import SimulationConfig
from src.stochastic.gbm import GeometricBrownianMotion
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AntitheticPricer:
    """
    Prices European options using antithetic variates variance reduction.

    Pairs each simulated path with its antithetic counterpart (-Z),
    averaging the two payoffs before discounting.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.gbm = GeometricBrownianMotion(config)

    def _payoff(self, prices: np.ndarray) -> np.ndarray:
        cfg = self.config
        if cfg.option_type == "call":
            return np.maximum(prices - cfg.K, 0)
        elif cfg.option_type == "put":
            return np.maximum(cfg.K - prices, 0)
        else:
            raise ValueError(f"Invalid option_type: '{cfg.option_type}'")

    def price(self, random_seed: int = None) -> dict:
        cfg = self.config
        half = cfg.n_simulations // 2

        terminal = self.gbm.get_terminal_prices_antithetic(random_seed=random_seed)

        terminal_original = terminal[:half]
        terminal_anti     = terminal[half:]

        payoff_original = self._payoff(terminal_original)
        payoff_anti     = self._payoff(terminal_anti)

        paired_payoffs = (payoff_original + payoff_anti) / 2

        discount = np.exp(-cfg.r * cfg.T)
        discounted = discount * paired_payoffs

        price     = discounted.mean()
        std_error = discounted.std() / np.sqrt(half)
        ci_lower  = price - 1.96 * std_error
        ci_upper  = price + 1.96 * std_error

        logger.info(f"Antithetic {cfg.option_type} price: {price:.6f}")
        logger.info(f"95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
        logger.info(f"Std Error: {std_error:.6f}")

        return {
            "price":     price,
            "std_error": std_error,
            "ci_lower":  ci_lower,
            "ci_upper":  ci_upper,
        }