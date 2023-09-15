#!/usr/bin/env python
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.core.prior import PriorDict, Uniform
from sampler_testing import run_test

np.random.seed(123)


class Hammer(Likelihood):
    def __init__(self):
        super().__init__(parameters={"x": None, "y": None})

    def log_likelihood(self):
        x = self.parameters["x"]
        y = self.parameters["y"]
        r = 1e-3
        scale = 30
        ret1 = -0.5 * (x**2 + y**2) / r**2
        ret2 = -0.5 * ((x / scale) ** 2 + (y * scale) ** 2) / r**2
        return ret1 * (x < 0) + ret2 * (x >= 0)


def draw_hammer(n_samples=100000):
    """
    Draw from a 2D Hammer distribution

    This is done by reparameterizing the problem as

    z = 10 (y - x^2)
    w = x - 1

    so that

    lnL = - (w^2 + z^2)
    """
    from scipy.stats import truncnorm

    n_below = sum(np.random.uniform(0, 1, n_samples) < 0.5)
    n_above = n_samples - n_below
    x_below = truncnorm(a=-np.inf, b=0, scale=1e-3).rvs(n_below)
    x_above = truncnorm(a=0, b=np.inf, scale=30e-3).rvs(n_above)
    y_below = truncnorm(a=-np.inf, b=np.inf, scale=1e-3).rvs(n_below)
    y_above = truncnorm(a=-np.inf, b=np.inf, scale=1e-3 / 30).rvs(n_above)
    x = np.concatenate([x_below, x_above])
    y = np.concatenate([y_below, y_above])
    return x, y


def simulate_expected_result(result):
    expected_result = deepcopy(result)
    expected_samples = draw_hammer(n_samples=len(result.posterior))
    expected_result.posterior["x"] = expected_samples[0]
    expected_result.posterior["y"] = expected_samples[1]
    expected_result.label = expected_result.label.replace("ensemble", "expected")
    expected_result.save_to_file(extension="hdf5")
    return expected_result


def plot_comparison(results):
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))
    for result in results:
        label = result.label.replace("_", " + ").title()
        posterior = result.posterior
        below_zero = posterior[posterior["x"] < 0]
        above_zero = posterior[posterior["x"] > 0]
        hist_kwargs = dict(bins=30, density=True, histtype="step", label=label)
        print(
            f'{label}: {np.mean(posterior["x"] < 0):.2f} +/- {len(posterior) ** -0.5:.2f}'
        )
        axes[0][0].hist(below_zero["x"], **hist_kwargs)
        axes[0][1].hist(above_zero["x"], **hist_kwargs)
        axes[1][0].hist(below_zero["y"], **hist_kwargs)
        axes[1][1].hist(above_zero["y"], **hist_kwargs)
    axes[0][0].legend(loc="upper left")
    axes[0][0].set_xlabel("$x$")
    axes[0][1].set_xlabel("$x$")
    axes[1][0].set_xlabel("$y$")
    axes[1][1].set_xlabel("$y$")
    axes[0][0].set_ylabel("$p(x | x < 0)$")
    axes[0][1].set_ylabel("$p(x | x \\geq 0)$")
    axes[1][0].set_ylabel("$p(y | x < 0)$")
    axes[1][1].set_ylabel("$p(y | x \\geq 0)$")
    plt.tight_layout()
    plt.savefig("outdir/hammer_comparison_posterior.pdf")
    plt.close()


if __name__ == "__main__":
    plt.style.use([Path(__file__).parent.parent / "paper.mplstyle"])
    priors = PriorDict()
    priors["x"] = Uniform(-1, 1, "x", latex_label="$x$")
    priors["y"] = Uniform(-1, 1, "y", latex_label="$y$")
    likelihood = Hammer()

    sampler = "dynesty"

    outdir = "outdir"
    label = "hammer"

    nlive = 100
    walks = 200

    results = run_test(
        likelihood=likelihood,
        priors=priors,
        outdir=outdir,
        label=label,
        nlive=nlive,
        walks=walks,
    )
    results.append(simulate_expected_result(results[0]))
    plot_comparison(results)
