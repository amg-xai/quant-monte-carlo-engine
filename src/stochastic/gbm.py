import numpy as np
from config.simulation_config import SimulationConfig


class GeometricBrownianMotion:
    """
    Simulates stock price paths using Geometric Brownian Motion.

    Under the risk-neutral measure:
        dS = r*S*dt + sigma*S*dW

    Discretized using the exact solution:
        S(t+dt) = S(t) * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        where Z ~ N(0,1)
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.dt = config.T / config.n_steps

    def simulate(self, random_seed: int = None) -> np.ndarray:
        if random_seed is not None:
            np.random.seed(random_seed)

        cfg = self.config
        dt = self.dt

        drift = (cfg.r - 0.5 * cfg.sigma ** 2) * dt
        Z = np.random.standard_normal((cfg.n_simulations, cfg.n_steps))
        diffusion = cfg.sigma * np.sqrt(dt) * Z
        log_returns = drift + diffusion

        paths = np.zeros((cfg.n_simulations, cfg.n_steps + 1))
        paths[:, 0] = cfg.S0
        paths[:, 1:] = cfg.S0 * np.exp(np.cumsum(log_returns, axis=1))

        return paths

    def get_terminal_prices(self, random_seed: int = None) -> np.ndarray:
        return self.simulate(random_seed=random_seed)[:, -1]
    
    def simulate_antithetic(self, random_seed: int = None) -> np.ndarray:
        """
        Generate price paths using antithetic variates.
        First half uses Z, second half uses -Z.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        cfg = self.config
        dt = self.dt
        half = cfg.n_simulations // 2

        drift = (cfg.r - 0.5 * cfg.sigma ** 2) * dt

        Z = np.random.standard_normal((half, cfg.n_steps))
        Z_anti = -Z
        Z_combined = np.concatenate([Z, Z_anti], axis=0)

        diffusion = cfg.sigma * np.sqrt(dt) * Z_combined
        log_returns = drift + diffusion

        paths = np.zeros((cfg.n_simulations, cfg.n_steps + 1))
        paths[:, 0] = cfg.S0
        paths[:, 1:] = cfg.S0 * np.exp(np.cumsum(log_returns, axis=1))

        return paths

    def get_terminal_prices_antithetic(self, random_seed: int = None) -> np.ndarray:
        """Returns terminal prices from antithetic simulation."""
        return self.simulate_antithetic(random_seed=random_seed)[:, -1]
    
