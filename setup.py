from setuptools import setup, find_packages

setup(
    name="quant-monte-carlo-engine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "plotly",
        "pytest",
        "yfinance",
    ],
)