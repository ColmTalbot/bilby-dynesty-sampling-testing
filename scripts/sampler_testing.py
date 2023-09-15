from copy import deepcopy
from os.path import isfile

import dill
import matplotlib.pyplot as plt
from bilby.core.result import plot_multiple, read_in_result
from bilby.core.sampler import run_sampler
from dynesty.utils import insertion_index_test


def comparison_stats_plot(sampler, fig=None, axs=None, color="blue", run_label=None):
    """
    Plot diagnostic statistics from a dynesty run

    The plotted quantities per iteration are:

    - nc: the number of likelihood calls
    - scale: the number of accepted MCMC steps if using :code:`bound="live"`
      or :code:`bound="live-multi"`, otherwise, the scale applied to the MCMC
      steps
    - lifetime: the number of iterations a point stays in the live set

    There is also a histogram of the lifetime compared with the theoretical
    distribution. To avoid edge effects, we discard the first 6 * nlive

    Parameters
    ----------
    sampler: dynesty.sampler.Sampler
        The sampler object containing the run history.

    Returns
    -------
    fig: matplotlib.pyplot.figure.Figure
        Figure handle for the new plot
    axs: matplotlib.pyplot.axes.Axes
        Axes handles for the new plot

    """

    if fig is None or axs is None:
        fig, axs = plt.subplots(nrows=2, figsize=(8, 8), sharex=True)
    # data = sampler.saved_run.D
    # for ax, name in zip(axs, ["nc", "scale"]):
    #     ax.plot(data[name], color=color)
    #     ax.set_ylabel(name.title())
    insertion_index_test(sampler.results, kind="likelihood", ax=axs[0])
    axs[0].set_xlabel("Insertion index / $n_{\\rm live}$")
    insertion_index_test(sampler.results, kind="distance", ax=axs[1])
    axs[1].set_xlabel("Insertion index / $n_{\\rm live}$")
    axs[1].set_xlim(0, 1)
    # lifetimes = np.arange(len(data["it"])) - data["it"]
    # axs[-2].set_ylabel("Lifetime")
    # if not hasattr(sampler, "nlive"):
    #     raise DynestySetupError("Cannot make stats plot for dynamic sampler.")
    # nlive = sampler.nlive
    # burn = int(geom(p=1 / nlive).isf(1 / 2 / nlive))
    # if len(data["it"]) > burn + sampler.nlive:
    #     axs[-2].plot(np.arange(0, burn), lifetimes[:burn], color=color, alpha=0.3)
    #     axs[-2].plot(
    #         np.arange(burn, len(lifetimes) - nlive),
    #         lifetimes[burn:-nlive],
    #         color=color,
    #         label=run_label,
    #     )
    #     axs[-2].plot(
    #         np.arange(len(lifetimes) - nlive, len(lifetimes)),
    #         lifetimes[-nlive:],
    #         color=color,
    #         alpha=0.5,
    #     )
    #     lifetimes = lifetimes[burn:-nlive]
    #     ks_result = ks_1samp(lifetimes, geom(p=1 / nlive).cdf)
    #     axs[-1].hist(
    #         lifetimes,
    #         bins=np.linspace(0, 6 * nlive, 60),
    #         histtype="step",
    #         density=True,
    #         color=color,
    #         label=f"p value = {ks_result.pvalue:.3f}",
    #     )
    #     axs[-1].plot(
    #         np.arange(1, 6 * nlive),
    #         geom(p=1 / nlive).pmf(np.arange(1, 6 * nlive)),
    #         color="red",
    #     )
    #     axs[-1].set_xlim(0, 6 * nlive)
    #     axs[-1].set_yscale("log")
    # else:
    #     axs[-2].plot(
    #         np.arange(0, len(lifetimes) - nlive),
    #         lifetimes[:-nlive],
    #         color=color,
    #         alpha=0.3,
    #     )
    #     axs[-2].plot(
    #         np.arange(len(lifetimes) - nlive, len(lifetimes)),
    #         lifetimes[-nlive:],
    #         color=color,
    #         alpha=0.5,
    #     )
    # axs[-2].set_yscale("log")
    # axs[-2].set_xlabel("Iteration")
    # axs[-1].set_xlabel("Lifetime")
    return fig, axs


def plot_corner_comparison(results, outdir="outdir", label="label"):
    plot_multiple(
        results,
        labels=["Ensemble", "Ellipsoid", "Ensemble + Ellipsoid", "Expected"],
        filename=f"{outdir}/{label}_comparison_corner.pdf",
        quantiles=None,
    )
    plt.close()


def plot_diagnostic_comparison(outdir="outdir", label="label"):
    fig = None
    for ii, run in enumerate(["ensemble", "ellipsoid", "ensemble_ellipsoid"]):
        if not isfile(f"{outdir}/{label}_{run}_resume.pickle"):
            continue
        with open(f"{outdir}/{label}_{run}_resume.pickle", "rb") as ff:
            kwargs = dict(color=f"C{ii}", run_label=run.replace("_", " + ").title())
            data = dill.load(ff)
            if fig is None:
                fig, axs = comparison_stats_plot(data, **kwargs)
            else:
                comparison_stats_plot(data, fig, axs, **kwargs)
    for ax in axs:
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/{label}_comparison_stats.pdf")
    plt.close()


def run_test(likelihood, priors, outdir="outdir", label="label", **kwargs):
    results = list()
    for ii, proposals in enumerate([["diff"], ["volumetric"], ["diff", "volumetric"]]):
        result = run_sampler(
            likelihood=deepcopy(likelihood),
            priors=deepcopy(priors),
            sampler="dynesty",
            outdir=outdir,
            label=f"{label}_{['ensemble', 'ellipsoid', 'ensemble_ellipsoid'][ii]}",
            maxmcmc=5000,
            check_point_delta_t=300,
            proposals=proposals,
            sample="acceptance-walk",
            nact=2,
            naccept=30,
            update_interval=10000,
            bound=["live", "live-multi", "live-multi"][ii],
            # first_update=dict(min_ncall=(walks - 1) / (1 - np.exp(-1 / nlive))),
            # ncdim=2,
            save="hdf5",
            **kwargs,
        )
        results.append(result)
    for run in ["dynesty", "expected"]:
        filename = f"{outdir}/{label}_{run}_result.hdf5"
        if isfile(filename):
            results.append(read_in_result(filename))

    if len(result.search_parameter_keys) < 10:
        plot_corner_comparison(results, outdir=outdir, label=label)
    plot_diagnostic_comparison(outdir=outdir, label=label)
    return results
