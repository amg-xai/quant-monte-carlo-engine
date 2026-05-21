import numpy as np
from config.simulation_config import SimulationConfig
from src.stochastic.gbm import GeometricBrownianMotion
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MonteCarloPricer:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.gbm = GeometricBrownianMotion(config)

    def _compute_payoffs(self, terminal_prices: np.ndarray) -> np.ndarray:
        cfg = self.config
        if cfg.option_type == "call":
            return np.maximum(terminal_prices - cfg.K, 0)
        elif cfg.option_type == "put":
            return np.maximum(cfg.K - terminal_prices, 0)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got '{cfg.option_type}'")

    def price(self, random_seed: int = None) -> dict:
        cfg = self.config

        terminal_prices = self.gbm.get_terminal_prices(random_seed=random_seed)
        payoffs = self._compute_payoffs(terminal_prices)

        discount_factor = np.exp(-cfg.r * cfg.T)
        discounted_payoffs = discount_factor * payoffs

        price = discounted_payoffs.mean()
        std_error = discounted_payoffs.std() / np.sqrt(cfg.n_simulations)

        ci_lower = price - 1.96 * std_error
        ci_upper = price + 1.96 * std_error

        logger.info(f"Monte Carlo {cfg.option_type} price: {price:.6f}")
        logger.info(f"95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
        logger.info(f"Std Error: {std_error:.6f}")

        return {
            "price": price,
            "std_error": std_error,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }