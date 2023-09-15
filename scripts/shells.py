#!/usr/bin/env python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.core.prior import PriorDict, Uniform
from sampler_testing import run_test

np.random.seed(123)


class Shell(Likelihood):
    def __init__(self, ndim):
        self.ndim = ndim
        self.centers = [np.zeros(ndim), np.zeros(ndim)]
        self.centers[0][0] = 3.5
        self.centers[1][0] = -3.5
        super().__init__(dict())

    def shell(self, point, centre, radius, scale):
        distance = np.linalg.norm(point - centre)
        return -((distance - radius) ** 2) / (2 * scale**2)

    def log_likelihood(self):
        point = np.array([self.parameters[f"x{ii}"] for ii in range(self.ndim)])
        return np.logaddexp(
            self.shell(point, self.centers[0], 2, 0.1),
            self.shell(point, self.centers[1], 2, 0.1),
        )


if __name__ == "__main__":
    plt.style.use([Path(__file__).parent.parent / "paper.mplstyle"])
    ndim = 2
    priors = PriorDict({f"x{ii}": Uniform(-6, 6) for ii in range(ndim)})
    likelihood = Shell(ndim)

    sampler = "dynesty"

    outdir = "outdir"
    label = f"shell_{ndim}"

    results = run_test(
        likelihood=likelihood,
        priors=priors,
        outdir=outdir,
        label=label,
        nlive=1000,
        walks=100,
    )
