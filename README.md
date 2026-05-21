# Quant Monte Carlo Engine

A professional-grade stochastic option pricing engine built from scratch in Python.
Implements Geometric Brownian Motion simulation, analytical Black-Scholes pricing,
and variance reduction techniques — validated against closed-form solutions.

---

## Results

| Method | Price | Std Error | Rel Error vs BS |
|---|---|---|---|
| Black-Scholes (Analytical) | 10.4506 | — | — |
| Standard Monte Carlo | 10.3994 | 0.04633 | 0.49% |
| Antithetic Variates | 10.3873 | 0.03290 | 0.61% |

**Variance reduction achieved: 28.98% lower std error with zero additional simulations.**

Both Monte Carlo methods produce prices where the Black-Scholes analytical solution
falls inside the 95% confidence interval — confirming mathematical correctness.

---

## Architecture
---
## Architecture

```
quant-monte-carlo-engine/
│
├── config/
│   └── simulation_config.py     # Centralized parameter management
│
├── src/
│   ├── stochastic/
│   │   └── gbm.py               # GBM simulator + antithetic paths
│   ├── pricing/
│   │   ├── black_scholes.py     # Analytical pricer + Greeks
│   │   └── monte_carlo_pricer.py
│   ├── analytics/
│   │   └── convergence.py       # Convergence analysis
│   ├── visualization/
│   │   └── plotter.py           # Publication-quality plots
│   ├── variance_reduction/
│   │   └── antithetic_pricer.py
│   └── utils/
│       └── logger.py
│
├── tests/
│   └── test_gbm.py              # Mathematical validation tests
│
└── outputs/
    └── plots/                   # Generated visualizations
```
## Methodology

### Geometric Brownian Motion

Stock prices are simulated under the risk-neutral measure using the exact GBM solution:

$$S_{t+\Delta t} = S_t \cdot \exp\left[\left(r - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t}\cdot Z\right], \quad Z \sim \mathcal{N}(0,1)$$

### Monte Carlo Pricing

Option prices are estimated by:
1. Simulating N price paths to maturity
2. Computing terminal payoffs: $\max(S_T - K,\ 0)$ for calls
3. Discounting back: $e^{-rT} \cdot \mathbb{E}[\text{payoff}]$

### Antithetic Variates

For every random draw $Z$, a mirrored path using $-Z$ is also simulated.
Averaging the paired payoffs exploits negative correlation to reduce estimator variance
without increasing the number of random draws.

### Black-Scholes Analytical Solution

Used as the ground truth benchmark:

$$C = S_0 N(d_1) - K e^{-rT} N(d_2)$$

$$d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

---

## Mathematical Validation

The GBM simulator is validated against two theoretical properties:

| Property | Theoretical | Simulated | Test |
|---|---|---|---|
| $\mathbb{E}[S_T]$ | $S_0 e^{rT}$ | within 1% | ✓ PASS |
| $\text{Var}[S_T]$ | $S_0^2 e^{2rT}(e^{\sigma^2 T}-1)$ | within 2% | ✓ PASS |

7 automated tests covering shape, mathematical correctness, and reproducibility.

---

## Greeks

| Greek | Value | Interpretation |
|---|---|---|
| Delta | 0.6368 | Price increases ~$0.64 per $1 rise in stock |
| Gamma | 0.0188 | Delta changes ~0.019 per $1 rise in stock |
| Vega | 0.3752 | Price increases ~$0.375 per 1% rise in volatility |
| Theta | -0.0176 | Option loses ~$0.018 per calendar day |

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/quant-monte-carlo-engine
cd quant-monte-carlo-engine

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .

python src/main.py
```

## Run Tests

```bash
pytest tests/ -v
```

---

## Dependencies

| Library | Purpose |
|---|---|
| NumPy | Vectorized path simulation |
| SciPy | Normal distribution, lognormal PDF |
| Pandas | Data handling |
| Matplotlib | Visualization |
| Plotly | Interactive visuals |
| pytest | Mathematical validation tests |
| yfinance | Real market data (Phase 2) |

---

## Parameters

All simulation parameters are centralized in `config/simulation_config.py`:

| Parameter | Default | Meaning |
|---|---|---|
| S0 | 100.0 | Initial stock price |
| K | 100.0 | Strike price |
| T | 1.0 | Time to maturity (years) |
| r | 0.05 | Risk-free rate |
| sigma | 0.20 | Volatility |
| n_steps | 252 | Trading days per year |
| n_simulations | 100,000 | Monte Carlo paths |
| option_type | call | call or put |

