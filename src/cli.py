import argparse
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quant Monte Carlo Engine — Professional Option Pricing"
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- Synthetic mode ---
    syn = subparsers.add_parser("synthetic", help="Price with custom parameters")
    syn.add_argument("--S0",          type=float, default=100.0)
    syn.add_argument("--K",           type=float, default=100.0)
    syn.add_argument("--T",           type=float, default=1.0)
    syn.add_argument("--r",           type=float, default=0.05)
    syn.add_argument("--sigma",       type=float, default=0.2)
    syn.add_argument("--n-steps",     type=int,   default=252)
    syn.add_argument("--n-sims",      type=int,   default=100_000)
    syn.add_argument("--option-type", type=str,   default="call",
                     choices=["call", "put"])

    # --- Market mode ---
    mkt = subparsers.add_parser("market", help="Price using live market data")
    mkt.add_argument("--ticker",      type=str,   required=True)
    mkt.add_argument("--option-type", type=str,   default="call",
                     choices=["call", "put"])

    return parser.parse_args()


def run_synthetic(args):
    from config.simulation_config import SimulationConfig
    from src.pricing.black_scholes import BlackScholesPricer
    from src.pricing.monte_carlo_pricer import MonteCarloPricer
    from src.variance_reduction.antithetic_pricer import AntitheticPricer
    from src.analytics.sensitivity import SensitivityAnalyzer
    from src.analytics.convergence import ConvergenceAnalyzer
    from src.visualization.plotter import Plotter
    import numpy as np

    config = SimulationConfig(
        S0=args.S0, K=args.K, T=args.T, r=args.r,
        sigma=args.sigma, n_steps=args.n_steps,
        n_simulations=args.n_sims,
        option_type=args.option_type
    )

    # Pricing
    bs   = BlackScholesPricer(config)
    mc   = MonteCarloPricer(config)
    anti = AntitheticPricer(config)

    bs_result   = bs.summary()
    mc_result   = mc.price(random_seed=42)
    anti_result = anti.price(random_seed=42)

    logger.info("=" * 55)
    logger.info(f"  BS    price : ${bs_result['price']:.6f}")
    logger.info(f"  MC    price : ${mc_result['price']:.6f}  SE={mc_result['std_error']:.6f}")
    logger.info(f"  Anti  price : ${anti_result['price']:.6f}  SE={anti_result['std_error']:.6f}")
    logger.info("=" * 55)

    # Sensitivity
    analyzer  = SensitivityAnalyzer(config)
    vol_df    = analyzer.sweep_volatility(np.linspace(0.05, 0.80, 50))
    strike_df = analyzer.sweep_strike(np.linspace(config.S0 * 0.7, config.S0 * 1.3, 50))
    mat_df    = analyzer.sweep_maturity(np.linspace(0.1, 2.0, 50))

    # Convergence
    conv         = ConvergenceAnalyzer(config)
    conv_results = conv.run([100, 500, 1000, 5000, 10000, 50000, 100000])

    # Plots
    plotter = Plotter(config)
    plotter.plot_all(results=conv_results, bs_price=bs_result["price"], random_seed=42)
    plotter.plot_sensitivity(vol_df, strike_df, mat_df)


def run_market(args):
    from src.market_pricing import run_market_pricing
    run_market_pricing(args.ticker, args.option_type)


def main():
    args = parse_args()
    if args.mode == "synthetic":
        run_synthetic(args)
    elif args.mode == "market":
        run_market(args)


if __name__ == "__main__":
    main()