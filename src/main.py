from config.simulation_config import SimulationConfig

def main():
    config = SimulationConfig()
    print("=" * 50)
    print("  Quant Monte Carlo Engine")
    print("=" * 50)
    print(f"  Asset:        S0 = {config.S0}")
    print(f"  Strike:       K  = {config.K}")
    print(f"  Maturity:     T  = {config.T} year(s)")
    print(f"  Risk-free:    r  = {config.r}")
    print(f"  Volatility:   σ  = {config.sigma}")
    print(f"  Steps:        {config.n_steps}")
    print(f"  Simulations:  {config.n_simulations:,}")
    print(f"  Option type:  {config.option_type}")
    print("=" * 50)
    print("  Configuration loaded. Engine ready.")
    print("=" * 50)

if __name__ == "__main__":
    main()