#!/usr/bin/env python
from copy import deepcopy
from pathlib import Path

import bilby
import matplotlib.pyplot as plt
import numpy as np
from sampler_testing import plot_corner_comparison, run_test

np.random.seed(123)


class Rosenbrock(bilby.Likelihood):
    def __init__(self):
        super().__init__(parameters={"x": None, "y": None})

    def log_likelihood(self):
        x = self.parameters["x"]
        y = self.parameters["y"]
        return -((1 - x) ** 2 + 100 * (y - x**2) ** 2)


def draw_rosenbrock(limit=5, n_samples=100000):
    """
    Draw from a 2D Rosenbrock distribution

    This is done by reparameterizing the problem as

    z = 10 (y - x^2)
    w = x - 1

    so that

    lnL = - (w^2 + z^2)
    """
    z = np.random.normal(0, 2**-0.5, n_samples * 2)
    w = np.random.normal(0, 2**-0.5, n_samples * 2)

    x = w + 1
    y = z / 10 + x**2

    keep = (abs(x) < limit) & (abs(y) < limit)

    x = x[keep][:n_samples]
    y = y[keep][:n_samples]
    return x, y


def simulate_expected_result(result):
    expected_result = deepcopy(result)
    expected_samples = draw_rosenbrock(limit=5, n_samples=len(result.posterior))
    expected_result.posterior["x"] = expected_samples[0]
    expected_result.posterior["y"] = expected_samples[1]
    expected_result.label = expected_result.label.replace("ensemble", "expected")
    expected_result.save_to_file(extension="hdf5")
    return expected_result


if __name__ == "__main__":
    plt.style.use([Path(__file__).parent.parent / "paper.mplstyle"])
    priors = bilby.core.prior.PriorDict()
    priors["x"] = bilby.core.prior.Uniform(-5, 5, "x", latex_label="$x$")
    priors["y"] = bilby.core.prior.Uniform(-5, 5, "y", latex_label="$y$")
    likelihood = Rosenbrock()

    sampler = "dynesty"

    outdir = "outdir"
    label = "rosenbrock"

    nlive = 5000
    walks = 100

    results = run_test(
        likelihood=likelihood,
        priors=priors,
        outdir=outdir,
        label=label,
        nlive=nlive,
        walks=walks,
    )
    results.append(simulate_expected_result(results[0]))
    plot_corner_comparison(results, outdir=outdir, label=label)
