from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from bilby.core.likelihood import AnalyticalMultidimensionalBimodalCovariantGaussian
from bilby.core.prior import PriorDict, Uniform
from sampler_testing import run_test


def analyze_result(result):
    log_prior_vol = np.sum(
        np.log([prior.maximum - prior.minimum for key, prior in result.priors.values()])
    )
    log_evidence = -log_prior_vol
    sampled_std_1 = []
    sampled_std_2 = []
    for param in result.search_parameter_keys:
        samples = np.array(result.posterior[param])
        samples_1 = samples[np.where(samples < 0)]
        samples_2 = samples[np.where(samples > 0)]
        sampled_std_1.append(np.std(samples_1))
        sampled_std_2.append(np.std(samples_2))

    print(f"Analytic log evidence:{log_evidence:.3f}")
    print(
        f"Sampled log evidence: {result.log_evidence:.3f} +/- {result.log_evidence_err:.3f}"
    )

    for i, search_parameter_key in enumerate(result.search_parameter_keys):
        print(search_parameter_key)
        print(f"Expected standard deviation both modes: {likelihood.sigma[i]}")
        print(f"Sampled standard deviation first mode: {sampled_std_1[i]}")
        print(f"Sampled standard deviation second mode: {sampled_std_2[i]}")


if __name__ == "__main__":
    plt.style.use([Path(__file__).parent.parent / "paper.mplstyle"])
    cov = np.loadtext(Path(__file__).parent / "covariance_matrix.dat")
    dim = len(cov[0])

    mean_1 = 4 * np.sqrt(np.diag(cov))
    mean_2 = -4 * np.sqrt(np.diag(cov))

    label = f"{dim}d_multidim_gaussian_bimodal"
    outdir = "outdir"

    likelihood = AnalyticalMultidimensionalBimodalCovariantGaussian(mean_1, mean_2, cov)
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
