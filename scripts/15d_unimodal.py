from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from bilby.core.likelihood import AnalyticalMultidimensionalCovariantGaussian
from bilby.core.prior import PriorDict, Uniform
from sampler_testing import run_test


def analyze_result(result):
    log_prior_vol = np.sum(
        np.log([prior.maximum - prior.minimum for prior in result.priors.values()])
    )
    log_evidence = -log_prior_vol

    sampled_std = result.posterior.std()

    print(f"Analytic log evidence: {log_evidence:.3f}")
    print(
        f"Sampled log evidence: {result.log_evidence:.3f} +/- {result.log_evidence_err:.3f}"
    )

    for i, key in enumerate(result.search_parameter_keys):
        print(key)
        print(f"Expected standard deviation: {likelihood.sigma[i]}")
        print(f"Sampled standard deviation: {sampled_std[key]}")


if __name__ == "__main__":
    plt.style.use([Path(__file__).parent.parent / "paper.mplstyle"])
    cov = np.loadtext(Path(__file__).parent / "covariance_matrix.dat")
    dim = len(cov[0])
    mean = np.zeros(dim)

    label = f"{dim}d_multidim_gaussian_unimodal"
    outdir = "outdir"

    likelihood = AnalyticalMultidimensionalCovariantGaussian(mean, cov)
    priors = PriorDict({f"x{i}": Uniform(-5, 5, f"x{i}") for i in range(dim)})

    results = run_test(
        likelihood=likelihood,
        priors=priors,
        outdir=outdir,
        label=label,
        nlive=500,
        walks=500,
    )

    for result in results:
        analyze_result(result)
