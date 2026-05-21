from dataclasses import dataclass

@dataclass
class SimulationConfig:
    # Asset parameters
    S0: float = 100.0        # Initial stock price
    K: float = 100.0         # Strike price
    T: float = 1.0           # Time to maturity (in years)
    r: float = 0.05          # Risk-free rate (5%)
    sigma: float = 0.2       # Volatility (20%)

    # Simulation parameters
    n_steps: int = 252        # Trading days in a year
    n_simulations: int = 10_000  # Number of Monte Carlo paths

    # Option type
    option_type: str = "call"  # "call" or "put"