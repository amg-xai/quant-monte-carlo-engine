import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import lognorm
from config.simulation_config import SimulationConfig
from src.stochastic.gbm import GeometricBrownianMotion
from src.utils.logger import get_logger
import os

logger = get_logger(__name__)

plt.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor": "#1a1a1a",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#cccccc",
    "axes.titlecolor": "#ffffff",
    "xtick.color": "#888888",
    "ytick.color": "#888888",
    "grid.color": "#2a2a2a",
    "grid.linestyle": "--",
    "text.color": "#cccccc",
    "font.family": "monospace",
})

ACCENT    = "#00ff88"
SECONDARY = "#ff6b35"
TERTIARY  = "#4fc3f7"
DIM       = "#444444"


class Plotter:

    def __init__(self, config: SimulationConfig, output_dir: str = "outputs/plots"):
        self.config = config
        self.output_dir = output_dir
        self.gbm = GeometricBrownianMotion(config)
        os.makedirs(output_dir, exist_ok=True)

    def plot_price_paths(self, n_display: int = 200, random_seed: int = 42):
        logger.info("Generating price paths plot...")
        paths = self.gbm.simulate(random_seed=random_seed)
        time_axis = np.linspace(0, self.config.T, self.config.n_steps + 1)

        fig, ax = plt.subplots(figsize=(12, 6))

        for i in range(min(n_display, self.config.n_simulations)):
            ax.plot(time_axis, paths[i], color=ACCENT, alpha=0.04, linewidth=0.6)

        mean_path = paths.mean(axis=0)
        ax.plot(time_axis, mean_path, color=SECONDARY, linewidth=2,
                label=f"Mean path: {mean_path[-1]:.2f}", zorder=5)

        ax.axhline(y=self.config.K, color=TERTIARY, linewidth=1.2,
                   linestyle="--", label=f"Strike K = {self.config.K}", zorder=4)

        ax.set_title("Simulated GBM Price Paths", fontsize=14, pad=15)
        ax.set_xlabel("Time (Years)")
        ax.set_ylabel("Stock Price ($)")
        ax.legend(loc="upper left", framealpha=0.2)
        ax.grid(True)

        plt.tight_layout()
        path = f"{self.output_dir}/price_paths.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {path}")

    def plot_terminal_distribution(self, random_seed: int = 42):
        logger.info("Generating terminal distribution plot...")
        terminal = self.gbm.get_terminal_prices(random_seed=random_seed)
        cfg = self.config

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.hist(terminal, bins=100, density=True, color=ACCENT,
                alpha=0.5, label="Simulated terminal prices")

        mu_ln = np.log(cfg.S0) + (cfg.r - 0.5 * cfg.sigma**2) * cfg.T
        sigma_ln = cfg.sigma * np.sqrt(cfg.T)
        x = np.linspace(terminal.min(), terminal.max(), 500)
        pdf = lognorm.pdf(x, s=sigma_ln, scale=np.exp(mu_ln))
        ax.plot(x, pdf, color=SECONDARY, linewidth=2, label="Theoretical lognormal")

        ax.axvline(x=cfg.K, color=TERTIARY, linewidth=1.2,
                   linestyle="--", label=f"Strike K = {cfg.K}")

        ax.set_title("Terminal Price Distribution", fontsize=14, pad=15)
        ax.set_xlabel("Terminal Stock Price ($)")
        ax.set_ylabel("Density")
        ax.legend(framealpha=0.2)
        ax.grid(True)

        plt.tight_layout()
        path = f"{self.output_dir}/terminal_distribution.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {path}")

    def plot_payoff_distribution(self, random_seed: int = 42):
        logger.info("Generating payoff distribution plot...")
        terminal = self.gbm.get_terminal_prices(random_seed=random_seed)
        cfg = self.config

        if cfg.option_type == "call":
            payoffs = np.maximum(terminal - cfg.K, 0)
        else:
            payoffs = np.maximum(cfg.K - terminal, 0)

        discount = np.exp(-cfg.r * cfg.T)
        discounted = discount * payoffs
        mc_price = discounted.mean()

        fig, ax = plt.subplots(figsize=(12, 6))

        nonzero = payoffs[payoffs > 0]
        ax.hist(nonzero, bins=80, density=True, color=ACCENT,
                alpha=0.6, label="Non-zero payoffs")

        zero_pct = (payoffs == 0).mean() * 100
        ax.axvline(x=0, color=DIM, linewidth=1)
        ax.axvline(x=mc_price, color=SECONDARY, linewidth=2,
                   linestyle="--", label=f"MC Price = {mc_price:.4f}")

        ax.set_title(
            f"Option Payoff Distribution  |  {zero_pct:.1f}% expire worthless",
            fontsize=14, pad=15
        )
        ax.set_xlabel("Payoff ($)")
        ax.set_ylabel("Density")
        ax.legend(framealpha=0.2)
        ax.grid(True)

        plt.tight_layout()
        path = f"{self.output_dir}/payoff_distribution.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {path}")

    def plot_convergence(self, results: dict, bs_price: float):
        """Plot MC vs antithetic convergence toward Black-Scholes price."""
        logger.info("Generating convergence plot...")

        sim_counts  = np.array(results["sim_counts"])
        mc_prices   = np.array(results["mc_prices"])
        mc_errors   = np.array(results["mc_errors"])
        anti_prices = np.array(results["anti_prices"])
        anti_errors = np.array(results["anti_errors"])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        ax1.semilogx(sim_counts, mc_prices, color=ACCENT, linewidth=1.5,
                     label="Standard MC")
        ax1.fill_between(sim_counts,
                         mc_prices - 1.96 * mc_errors,
                         mc_prices + 1.96 * mc_errors,
                         color=ACCENT, alpha=0.15, label="MC 95% CI")

        ax1.semilogx(sim_counts, anti_prices, color=SECONDARY, linewidth=1.5,
                     label="Antithetic")
        ax1.fill_between(sim_counts,
                         anti_prices - 1.96 * anti_errors,
                         anti_prices + 1.96 * anti_errors,
                         color=SECONDARY, alpha=0.15, label="Antithetic 95% CI")

        ax1.axhline(y=bs_price, color=TERTIARY, linewidth=1.5,
                    linestyle="--", label=f"Black-Scholes = {bs_price:.4f}")

        ax1.set_title("Price Convergence vs Simulations", fontsize=13, pad=12)
        ax1.set_ylabel("Estimated Price ($)")
        ax1.legend(framealpha=0.2, fontsize=9)
        ax1.grid(True)

        ax2.loglog(sim_counts, mc_errors, color=ACCENT, linewidth=1.5,
                   label="Standard MC std error")
        ax2.loglog(sim_counts, anti_errors, color=SECONDARY, linewidth=1.5,
                   label="Antithetic std error")

        ref = mc_errors[0] * np.sqrt(sim_counts[0]) / np.sqrt(sim_counts)
        ax2.loglog(sim_counts, ref, color=DIM, linewidth=1,
                   linestyle=":", label="1/√N reference")

        ax2.set_title("Std Error Convergence (log-log)", fontsize=13, pad=12)
        ax2.set_xlabel("Number of Simulations")
        ax2.set_ylabel("Std Error")
        ax2.legend(framealpha=0.2, fontsize=9)
        ax2.grid(True)

        plt.tight_layout()
        path = f"{self.output_dir}/convergence.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {path}")

    def plot_volatility_surface(self, surface_data: list, spot_price: float, ticker: str = ""):
        """Plot implied volatility surface across strikes and expiries."""
        logger.info("Generating volatility surface plot...")

        import pandas as pd

        df = pd.DataFrame(surface_data)
        df = df.dropna(subset=["iv"])
        calls = df[df["option_type"] == "call"]

        pivot = calls.pivot_table(index="strike", columns="expiry", values="iv")
        pivot = pivot.dropna(how="all").ffill(axis=1)

        strikes  = pivot.index.values
        expiries = np.arange(len(pivot.columns))
        Z        = pivot.values * 100

        fig = plt.figure(figsize=(14, 8))
        ax  = fig.add_subplot(111, projection="3d")

        X, Y = np.meshgrid(expiries, strikes)

        surf = ax.plot_surface(X, Y, Z, cmap="plasma", alpha=0.85, linewidth=0)
        fig.colorbar(surf, ax=ax, shrink=0.4, label="Implied Vol (%)")

        ax.set_xlabel("Expiry")
        ax.set_ylabel("Strike ($)")
        ax.set_zlabel("IV (%)")
        ax.set_title(f"Volatility Surface — {ticker}", fontsize=13, pad=20)

        ax.set_xticks(expiries)
        ax.set_xticklabels(pivot.columns, rotation=45, fontsize=6)

        ax.set_facecolor("#1a1a1a")
        fig.patch.set_facecolor("#0f0f0f")
        ax.tick_params(colors="#888888")

        plt.tight_layout()
        path = f"{self.output_dir}/volatility_surface.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {path}")

    def plot_sensitivity(self, vol_df, strike_df, maturity_df):
        """Plot option price sensitivity to vol, strike, and maturity."""
        logger.info("Generating sensitivity analysis plot...")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

        # Vol sensitivity
        ax1.plot(vol_df["sigma"] * 100, vol_df["price"],
                 color=ACCENT, linewidth=2)
        ax1.axvline(x=self.config.sigma * 100, color=SECONDARY,
                    linestyle="--", linewidth=1.2, label=f"Current σ = {self.config.sigma*100:.0f}%")
        ax1.set_title("Price vs Volatility", fontsize=12)
        ax1.set_xlabel("Volatility (%)")
        ax1.set_ylabel("Option Price ($)")
        ax1.legend(framealpha=0.2)
        ax1.grid(True)

        # Strike sensitivity
        ax2.plot(strike_df["strike"], strike_df["price"],
                 color=ACCENT, linewidth=2)
        ax2.axvline(x=self.config.S0, color=TERTIARY,
                    linestyle="--", linewidth=1.2, label=f"Spot = ${self.config.S0:.0f}")
        ax2.axvline(x=self.config.K, color=SECONDARY,
                    linestyle="--", linewidth=1.2, label=f"Strike = ${self.config.K:.0f}")
        ax2.set_title("Price vs Strike", fontsize=12)
        ax2.set_xlabel("Strike ($)")
        ax2.set_ylabel("Option Price ($)")
        ax2.legend(framealpha=0.2)
        ax2.grid(True)

        # Maturity sensitivity
        ax3.plot(maturity_df["maturity"], maturity_df["price"],
                 color=ACCENT, linewidth=2)
        ax3.axvline(x=self.config.T, color=SECONDARY,
                    linestyle="--", linewidth=1.2, label=f"Current T = {self.config.T:.2f}y")
        ax3.set_title("Price vs Maturity", fontsize=12)
        ax3.set_xlabel("Time to Maturity (Years)")
        ax3.set_ylabel("Option Price ($)")
        ax3.legend(framealpha=0.2)
        ax3.grid(True)

        plt.suptitle("Sensitivity Analysis", fontsize=14, y=1.02)
        plt.tight_layout()
        path = f"{self.output_dir}/sensitivity.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {path}")



    def plot_volatility_smile(self, chain, spot_price: float, title: str = ""):
        """Plot implied volatility smile/skew from real options chain."""
        logger.info("Generating volatility smile plot...")

        import pandas as pd

        calls = chain[chain["option_type"] == "call"].dropna(subset=["computed_iv"])
        puts  = chain[chain["option_type"] == "put"].dropna(subset=["computed_iv"])

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(calls["strike"], calls["computed_iv"] * 100,
                color=ACCENT, linewidth=2, marker="o", markersize=3, label="Call IV")
        ax.plot(puts["strike"], puts["computed_iv"] * 100,
                color=SECONDARY, linewidth=2, marker="o", markersize=3, label="Put IV")

        ax.axvline(x=spot_price, color=TERTIARY, linewidth=1.5,
                   linestyle="--", label=f"Spot = ${spot_price:.2f}")

        ax.set_title(f"Implied Volatility Smile — {title}", fontsize=14, pad=15)
        ax.set_xlabel("Strike Price ($)")
        ax.set_ylabel("Implied Volatility (%)")
        ax.legend(framealpha=0.2)
        ax.grid(True)

        plt.tight_layout()
        path = f"{self.output_dir}/volatility_smile.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {path}")

        

    def plot_all(self, results: dict = None, bs_price: float = None, random_seed: int = 42):
        self.plot_price_paths(random_seed=random_seed)
        self.plot_terminal_distribution(random_seed=random_seed)
        self.plot_payoff_distribution(random_seed=random_seed)
        if results is not None and bs_price is not None:
            self.plot_convergence(results, bs_price)
        logger.info("All plots saved to outputs/plots/")