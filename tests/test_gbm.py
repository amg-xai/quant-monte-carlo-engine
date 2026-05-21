import numpy as np
import pytest
from config.simulation_config import SimulationConfig
from src.stochastic.gbm import GeometricBrownianMotion


@pytest.fixture
def default_config():
    """Standard config used across tests."""
    return SimulationConfig(
        S0=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        n_steps=252,
        n_simulations=100_000,
        option_type="call"
    )


class TestGBMShape:

    def test_path_matrix_shape(self, default_config):
        gbm = GeometricBrownianMotion(default_config)
        paths = gbm.simulate(random_seed=42)
        expected_shape = (default_config.n_simulations, default_config.n_steps + 1)
        assert paths.shape == expected_shape

    def test_all_paths_start_at_S0(self, default_config):
        gbm = GeometricBrownianMotion(default_config)
        paths = gbm.simulate(random_seed=42)
        assert np.allclose(paths[:, 0], default_config.S0)

    def test_no_negative_prices(self, default_config):
        gbm = GeometricBrownianMotion(default_config)
        paths = gbm.simulate(random_seed=42)
        assert np.all(paths > 0)


class TestGBMMath:

    def test_expected_terminal_price(self, default_config):
        gbm = GeometricBrownianMotion(default_config)
        terminal = gbm.get_terminal_prices(random_seed=42)
        simulated_mean = terminal.mean()
        theoretical_mean = default_config.S0 * np.exp(default_config.r * default_config.T)
        relative_error = abs(simulated_mean - theoretical_mean) / theoretical_mean
        assert relative_error < 0.01

    def test_expected_variance(self, default_config):
        gbm = GeometricBrownianMotion(default_config)
        terminal = gbm.get_terminal_prices(random_seed=42)
        simulated_var = terminal.var()
        cfg = default_config
        theoretical_var = (cfg.S0 ** 2 * np.exp(2 * cfg.r * cfg.T) * (np.exp(cfg.sigma ** 2 * cfg.T) - 1))
        relative_error = abs(simulated_var - theoretical_var) / theoretical_var
        assert relative_error < 0.02


class TestGBMReproducibility:

    def test_same_seed_same_paths(self, default_config):
        gbm = GeometricBrownianMotion(default_config)
        paths_1 = gbm.simulate(random_seed=42)
        paths_2 = gbm.simulate(random_seed=42)
        assert np.array_equal(paths_1, paths_2)

    def test_different_seeds_different_paths(self, default_config):
        gbm = GeometricBrownianMotion(default_config)
        paths_1 = gbm.simulate(random_seed=42)
        paths_2 = gbm.simulate(random_seed=99)
        assert not np.array_equal(paths_1, paths_2)