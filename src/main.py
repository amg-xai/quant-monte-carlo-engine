from config.simulation_config import SimulationConfig
from src.pricing.black_scholes import BlackScholesPricer
from src.pricing.monte_carlo_pricer import MonteCarloPricer
from src.utils.logger import get_logger

logger = get_logger(__name__)


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

    # --- Black-Scholes ---
    bs = BlackScholesPricer(config)
    bs_summary = bs.summary()

    logger.info("BLACK-SCHOLES (Analytical)")
    logger.info(f"  Price : {bs_summary['price']:.6f}")
    logger.info(f"  Delta : {bs_summary['delta']:.6f}")
    logger.info(f"  Gamma : {bs_summary['gamma']:.6f}")
    logger.info(f"  Vega  : {bs_summary['vega']:.6f}")
    logger.info(f"  Theta : {bs_summary['theta']:.6f}")

    logger.info("-" * 55)

    # --- Monte Carlo ---
    mc = MonteCarloPricer(config)
    mc_result = mc.price(random_seed=42)

    logger.info("MONTE CARLO (Simulation)")
    logger.info(f"  Price     : {mc_result['price']:.6f}")
    logger.info(f"  Std Error : {mc_result['std_error']:.6f}")
    logger.info(f"  95% CI    : [{mc_result['ci_lower']:.6f}, {mc_result['ci_upper']:.6f}]")

    logger.info("-" * 55)

    # --- Comparison ---
    diff = abs(bs_summary['price'] - mc_result['price'])
    rel_error = diff / bs_summary['price'] * 100

    logger.info("COMPARISON")
    logger.info(f"  Absolute difference : {diff:.6f}")
    logger.info(f"  Relative error      : {rel_error:.4f}%")
    logger.info(f"  BS inside MC 95% CI : "
                f"{'YES' if mc_result['ci_lower'] <= bs_summary['price'] <= mc_result['ci_upper'] else 'NO'}")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()