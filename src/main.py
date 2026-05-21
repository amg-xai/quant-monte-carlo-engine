from config.simulation_config import SimulationConfig
from src.stochastic.gbm import GeometricBrownianMotion
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    logger.info("Initializing simulation engine...")

    config = SimulationConfig()
    gbm = GeometricBrownianMotion(config)

    logger.info(f"Running {config.n_simulations:,} simulations over {config.n_steps} steps...")

    paths = gbm.simulate(random_seed=42)

    logger.info(f"Simulation complete.")
    logger.info(f"Path matrix shape: {paths.shape}")
    logger.info(f"Starting price (all paths): {paths[:, 0].mean():.2f}")
    logger.info(f"Mean terminal price:        {paths[:, -1].mean():.2f}")
    logger.info(f"Min terminal price:         {paths[:, -1].min():.2f}")
    logger.info(f"Max terminal price:         {paths[:, -1].max():.2f}")

if __name__ == "__main__":
    main()