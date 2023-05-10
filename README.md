# Stein-Pi-Importance-Sampling

This code accompanies the paper "Stein $\Pi$-Importance Sampling". It was written in Python 3.10.4 and also tested in Python 3.10.4.

This package can be installed directly by pip in the root directory

```bash
python -m pip install .
```

Some of the experiments in this paper rely on the following packages

- Stein Thinning
- [posteriordb v0.4.0](https://github.com/stan-dev/posteriordb)
- [posteriordb-python](https://github.com/stan-dev/posteriordb-python)
- [BridgeStan v1.0.2](https://roualdes.github.io/bridgestan/latest/)
- [qpsolvers v3.4.0](https://github.com/qpsolvers/qpsolvers)
- [proxsuite v0.3.7](https://github.com/Simple-Robotics/proxsuite)
- [Wasserstein v1.1.0](https://github.com/pkomiske/Wasserstein/)

The original package for Stein Thinning has been modified, and it can be installed in the following way

```bash
cd stein_thinning
python -m pip install .
```

To reproduce the experiment, navigate to "experiment" and run the relevant script.

e.g. "gene_store_wksd.py" will run the benchmarking on PosteriorDB to store the numpy data files. "plot_KSDCurve.py" will plot the corresponding results.
