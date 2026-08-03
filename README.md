# quasi_cem

This repository implements sampling and optimization routines using the Cross Entropy Method (CEM) and Sobol sequence sampling via SciPy.

## Methods

- **Cross Entropy Method (CEM):**
	- Used for optimization and sampling in high-dimensional spaces.
	- Reference: [Rubinstein, R.Y., & Kroese, D.P. (2004). The Cross-Entropy Method: A Unified Approach to Combinatorial Optimization, Monte-Carlo Simulation, and Machine Learning. Springer.](https://link.springer.com/book/10.1007/978-1-4419-1643-7)

- **Sobol Sampling (SciPy):**
	- Quasi-random low-discrepancy sequence for efficient sampling.
	- Reference: [SciPy Documentation - scipy.stats.qmc.Sobol](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html)

## Usage

See `data_sampler.py` for implementation details

## Interface Use Case

For a generic tuning interface example, see:
[arbok_driver/generic_tunig_interface.py](https://github.com/andncl/arbok_driver/blob/master/arbok_driver/generic_tuning_interface.py)
