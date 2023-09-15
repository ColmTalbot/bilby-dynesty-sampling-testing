#!/usr/bin/env python
"""
A script to demonstrate how to use your own source model
"""
from pathlib import Path

import bilby
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.core.prior import LogUniform, PriorDict, Sine, Uniform
from bilby.gw.conversion import _generate_all_cbc_parameters
from bilby.gw.likelihood import GravitationalWaveTransient
from sampler_testing import run_test


def burst_conversion(parameters: dict):
    output = parameters.copy()
    added_keys = list()
    if "hrss" not in parameters and all(
        key in parameters for key in ["luminosity_distance", "amplitude"]
    ):
        output["hrss"] = parameters["amplitude"] / parameters["luminosity_distance"]
        added_keys.append("hrss")
    return output, added_keys


def burst_generation(
    sample: dict,
    likelihood: Likelihood = None,
    priors: PriorDict = None,
    npool: int = 1,
):
    return _generate_all_cbc_parameters(
        sample,
        defaults=dict(),
        base_conversion=burst_conversion,
        likelihood=likelihood,
        priors=priors,
        npool=npool,
    )


def burst_sine_gaussian(
    frequency_array: np.ndarray,
    hrss: float,
    quality: float,
    frequency: float,
    phase: float,
    eccentricity: float,
):
    """
    Translation of lalinference.BurstSineGaussianF with the following changes:
    - The phase is adapted to match the convention for CBCs to allow marginalized phase reconstruction.
    - Luminsity distance is added as a parameter to allow for distance marginalization.

    Parameters
    ----------
    frequency_array: array_like
        The frequency array in Hz
    hrss: float
        The amplitude of the signal at 1 Mpc
    quality: float
        The quality factor of the signal
    frequency: float
        The central frequency of the signal
    phase: float
        The constant phase of the signal at zero time
    eccentricity: float
        The eccentricity of the signal

    Returns
    -------
    dict: The plus and cross polarizations of the signal
    """
    hplus = np.zeros(len(frequency_array), dtype=complex)
    hcross = np.zeros(len(frequency_array), dtype=complex)

    semi_major = (2 - eccentricity**2) ** -0.5
    semi_minor = semi_major * (1 - eccentricity**2) ** 0.5
    sigma = frequency / quality
    # cos(2x) = cos(x)^2 - sin(x)^2
    decay = np.exp(-(quality**2)) * np.cos(2 * phase)

    hrss /= np.pi**0.25 * 2**0.5 * sigma**0.5

    width = 6 * sigma
    mask = abs(frequency_array - frequency) <= width

    delta_f = frequency_array[mask] - frequency
    envelope = np.exp(-(delta_f**2) / (2 * sigma**2) + 2j * phase)
    hplus[mask] = hrss * semi_major / (1 + decay) ** 0.5 * envelope
    hcross[mask] = -1j * hrss * semi_minor / (1 - decay) ** 0.5 * envelope
    return dict(plus=hplus, cross=hcross)


class Circular(Uniform):
    def __init__(
        self,
        period: float = 2 * np.pi,
        name: str = None,
        latex_label: str = None,
        unit: str = None,
    ):
        """
        A uniform prior on a circular parameter

        Parameters
        ----------
        period: float
            The period of the parameter, defaults to 2 pi
        """
        self.period = period
        super().__init__(
            minimum=0,
            maximum=period,
            name=name,
            latex_label=latex_label,
            unit=unit,
            boundary="periodic",
        )


def fetch_open_data(event_name, duration, psd_duration=None, detectors=None):
    from glob import glob

    from gwosc.datasets import event_gps
    from gwpy.timeseries import TimeSeries, TimeSeriesDict
    from requests.exceptions import ConnectionError

    if detectors is None:
        detectors = ["H1", "L1", "V1"]

    try:
        gps = event_gps(event_name)
    except ConnectionError:
        return TimeSeriesDict.read(glob(f"{event_name}*.hdf5"))
    start = gps - duration + 2
    end = gps + 2
    if psd_duration is not None:
        start -= psd_duration

    filename = f"{event_name}_{''.join(detectors)}_{start}_{int(end - start)}.hdf5"
    if Path(filename).exists():
        return TimeSeriesDict.read(filename)

    data = TimeSeriesDict()
    for detector in detectors:
        data[detector] = TimeSeries.fetch_open_data(
            detector, start, end, verbose=True, sample_rate=4096
        )
    data.write(filename)
    return data


def strain_data_to_interferometers(data_dict: dict, duration: int):
    from bilby.gw.detector import InterferometerList, PowerSpectralDensity

    ifos = list()
    for key, data in data_dict.items():
        times = data.times.value
        analysis_data = data[times - times[-1] > -duration]
        psd_data = data[times - times[-1] <= -duration]
        # breakpoint()
        ifo = bilby.gw.detector.get_empty_interferometer(key)
        ifo.set_strain_data_from_gwpy_timeseries(analysis_data)
        # psd_data = psd_data.gate()
        psd = psd_data.psd(
            fftlength=ifo.duration,
            overlap=0,
            window=("tukey", ifo.strain_data.alpha),
            method="median",
        )
        ifo.power_spectral_density = PowerSpectralDensity(
            frequency_array=psd.frequencies.value, psd_array=psd.value
        )
        ifos.append(ifo)

    return InterferometerList(ifos)


def get_burst_prior(trigger_time):
    priors = PriorDict()
    priors["frequency"] = Uniform(30, 360, latex_label="$f_{0}$", boundary="reflective")
    priors["quality"] = Uniform(3, 108, latex_label="$Q$", boundary="reflective")
    priors["phase"] = Circular(latex_label="$\\phi$")
    priors["zenith"] = Sine(latex_label="$\\delta$")
    priors["azimuth"] = Circular(latex_label="$\\alpha$")
    priors["psi"] = Circular(np.pi, latex_label="$\\psi$")
    priors["L1_time"] = Uniform(
        ifos.start_time + 1.9, ifos.start_time + 2.1, latex_label="$t_{L}$"
    )
    priors["eccentricity"] = Uniform(0, 1, boundary="reflective")
    priors["amplitude"] = 1e-18
    priors["luminosity_distance"] = LogUniform(1, 1e6, latex_label="$d_{L}$")
    return priors


def get_and_plot_data(event_name, duration, outdir, label):
    data = fetch_open_data(
        event_name=event_name,
        duration=duration,
        psd_duration=128,
        detectors=["H1", "L1"],
    )
    ifos = strain_data_to_interferometers(data, duration=duration)
    for ifo in ifos:
        # ifo.minimum_frequency = 70
        ifo.maximum_frequency = 400
    ifos.plot_data(outdir=outdir, label=event_name)
    return ifos


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("fork")

    plt.style.use([Path(__file__).parent.parent / "paper.mplstyle"])

    outdir = "outdir"
    label = "burst"
    event_name = "GW200129_065458"
    duration = 4
    np.random.seed(400)
    bilby.core.utils.random.seed(400)

    ifos = get_and_plot_data(event_name, duration, outdir, label)

    priors = get_burst_prior(ifos.start_time + 2)

    waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration,
        sampling_frequency=ifos.sampling_frequency,
        frequency_domain_source_model=burst_sine_gaussian,
        parameter_conversion=burst_conversion,
    )

    likelihood = GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=waveform_generator,
        reference_frame="H1L1",
        time_reference="L1",
        phase_marginalization=True,
        priors=priors,
        distance_marginalization=True,
        distance_marginalization_lookup_table=f"{label}_lookup.npz",
    )

    nlive = 500

    results = run_test(
        likelihood=likelihood,
        priors=priors,
        outdir=outdir,
        label=label,
        nlive=nlive,
        conversion_function=burst_generation,
        result_class=bilby.gw.result.CBCResult,
        npool=4,
    )


def make_plots(result):
    result.plot_waveform_posterior(
        format="pdf",
        interferometers=ifos,
        n_samples=500,
    )

    result.plot_corner(
        parameters=[
            "frequency",
            "quality",
            "eccentricity",
            "L1_time",
            "psi",
            "azimuth",
            "zenith",
            "phase",
            "hrss",
        ]
    )

    # results = list()
    # for ii, proposals in enumerate([["diff", "stretch"], ["normal"], ["diff", "stretch", "normal"]]):
    #     result = bilby.run_sampler(
    #         likelihood=likelihood, priors=priors, sampler='dynesty', nlive=100,
    #         injection_parameters=injection_parameters, outdir=outdir,
    #         label=f"{label}_{['ensemble', 'ellipsoid', 'ensemble_ellipsoid'][ii]}",
    #         maxmcmc=10000,
    #         check_point_delta_t=300,
    #         proposals=proposals,
    #         first_update=dict(min_eff=5, min_ncall=400),
    #         walks=500, nact=20,
    #         queue_size=1,
    #     )
    #     converted = {key: list() for key in ["distance", "ra", "dec", "geocent_time"][1:]}
    #     for ii in trange(len(result.posterior)):
    #         parameters = dict(result.posterior.iloc[ii])
    #         likelihood.parameters = parameters
    #         sky_frame = likelihood.get_sky_frame_parameters()
    #         for key in sky_frame:
    #             converted[key].append(sky_frame[key])
    #     for key in converted:
    #         converted[key] = np.array(converted[key])
    #         result.posterior[key] = converted[key]
    #     results.append(result)
    #
    # bilby.core.result.plot_multiple(
    #     results, labels=["Ensemble", "Ellipsoid", "Ensemble + Ellipsoid"],
    #     filename="outdir/burst_comparison_corner.pdf"
    # )
    #
    # fig = None
    # for ii, run in enumerate(['ensemble', 'ellipsoid', 'ensemble_ellipsoid']):
    #     with open(f"outdir/{label}_{run}_resume.pickle", "rb") as ff:
    #         data = dill.load(ff)
    #         if fig is None:
    #             fig, axs = bilby.core.sampler.dynesty.dynesty_stats_plot(
    #                 data, color=f"C{ii}", run_label=run.replace("_", " + ").title()
    #             )
    #         else:
    #             bilby.core.sampler.dynesty.dynesty_stats_plot(
    #                 data, fig, axs, color=f"C{ii}", run_label=run.replace("_", " + ").title()
    #             )
    # plt.tight_layout()
    # plt.savefig("outdir/burst_comparison_stats.pdf")
    # plt.close()
