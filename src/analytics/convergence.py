import numpy as np
from config.simulation_config import SimulationConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConvergenceAnalyzer:
    """
    Analyzes how Monte Carlo price estimates converge
    to the true Black-Scholes value as n_simulations increases.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def _payoff(self, terminal: np.ndarray) -> np.ndarray:
        cfg = self.config
        if cfg.option_type == "call":
            return np.maximum(terminal - cfg.K, 0)
        return np.maximum(cfg.K - terminal, 0)

    def run(self, sim_counts: list, random_seed: int = 42) -> dict:
        logger.info(f"Running convergence analysis over {len(sim_counts)} sample sizes...")

        cfg = self.config
        discount = np.exp(-cfg.r * cfg.T)
        dt = cfg.T / cfg.n_steps
        drift = (cfg.r - 0.5 * cfg.sigma ** 2) * dt

        max_sims = max(sim_counts)
        np.random.seed(random_seed)
        Z_full = np.random.standard_normal((max_sims, cfg.n_steps))

        mc_prices, mc_errors = [], []
        anti_prices, anti_errors = [], []

        for n in sim_counts:
            # --- Standard MC ---
            Z = Z_full[:n]
            log_returns = drift + cfg.sigma * np.sqrt(dt) * Z
            terminal = cfg.S0 * np.exp(np.sum(log_returns, axis=1))
            payoffs = discount * self._payoff(terminal)
            mc_prices.append(payoffs.mean())
            mc_errors.append(payoffs.std() / np.sqrt(n))

            # --- Antithetic ---
            half = n // 2
            Z_half = Z_full[:half]
            Z_anti = np.concatenate([Z_half, -Z_half], axis=0)
            log_returns_a = drift + cfg.sigma * np.sqrt(dt) * Z_anti
            terminal_a = cfg.S0 * np.exp(np.sum(log_returns_a, axis=1))
            payoffs_orig = discount * self._payoff(terminal_a[:half])
            payoffs_anti = discount * self._payoff(terminal_a[half:])
            paired = (payoffs_orig + payoffs_anti) / 2
            anti_prices.append(paired.mean())
            anti_errors.append(paired.std() / np.sqrt(half))

        logger.info("Convergence analysis complete.")

        return {
            "sim_counts":  sim_counts,
            "mc_prices":   mc_prices,
            "mc_errors":   mc_errors,
            "anti_prices": anti_prices,
            "anti_errors": anti_errors,
        }