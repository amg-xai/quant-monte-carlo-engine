from config.simulation_config import SimulationConfig
from src.pricing.black_scholes import BlackScholesPricer
from src.pricing.monte_carlo_pricer import MonteCarloPricer
from src.variance_reduction.antithetic_pricer import AntitheticPricer
from src.visualization.plotter import Plotter
from src.utils.logger import get_logger

logger = get_logger(__name__)


def print_result(label: str, result: dict, bs_price: float):
    rel_error = abs(bs_price - result["price"]) / bs_price * 100
    inside_ci = result["ci_lower"] <= bs_price <= result["ci_upper"]
    logger.info(f"{label}")
    logger.info(f"  Price     : {result['price']:.6f}")
    logger.info(f"  Std Error : {result['std_error']:.6f}")
    logger.info(f"  95% CI    : [{result['ci_lower']:.6f}, {result['ci_upper']:.6f}]")
    logger.info(f"  Rel Error : {rel_error:.4f}%")
    logger.info(f"  BS in CI  : {'YES' if inside_ci else 'NO'}")


def main():
    config = SimulationConfig(
        S0=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        n_steps=252,
        n_simulations=100_000,
        option_type="call"
    )

    logger.info("=" * 55)
    logger.info("  QUANT MONTE CARLO ENGINE")
    logger.info("=" * 55)

    # Black-Scholes benchmark
    bs = BlackScholesPricer(config)
    bs_summary = bs.summary()
    logger.info("BLACK-SCHOLES (Analytical Benchmark)")
    logger.info(f"  Price : {bs_summary['price']:.6f}")
    logger.info(f"  Delta : {bs_summary['delta']:.6f}")
    logger.info(f"  Gamma : {bs_summary['gamma']:.6f}")
    logger.info(f"  Vega  : {bs_summary['vega']:.6f}")
    logger.info(f"  Theta : {bs_summary['theta']:.6f}")
    logger.info("-" * 55)

    # Standard Monte Carlo
    mc = MonteCarloPricer(config)
    mc_result = mc.price(random_seed=42)
    print_result("STANDARD MONTE CARLO", mc_result, bs_summary["price"])
    logger.info("-" * 55)

    # Antithetic variates
    anti = AntitheticPricer(config)
    anti_result = anti.price(random_seed=42)
    print_result("ANTITHETIC VARIATES", anti_result, bs_summary["price"])
    logger.info("-" * 55)

    # Variance reduction summary
    std_reduction = (1 - anti_result["std_error"] / mc_result["std_error"]) * 100
    logger.info("VARIANCE REDUCTION SUMMARY")
    logger.info(f"  Standard MC std error  : {mc_result['std_error']:.6f}")
    logger.info(f"  Antithetic std error   : {anti_result['std_error']:.6f}")
    logger.info(f"  Std error reduction    : {std_reduction:.2f}%")
    logger.info("=" * 55)

    # Plots
    plotter = Plotter(config)
    plotter.plot_all(random_seed=42)


if __name__ == "__main__":
    main()