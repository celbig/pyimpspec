# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# Copyright 2026 pyimpspec developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# The licenses of pyimpspec's dependencies and/or sources of portions of code are included in
# the LICENSES folder.

# This module uses the hierarchical Bayesian inversion implemented by the
# optional bayes_drt2 package (a cmdstanpy-based re-implementation of bayes_drt)
# - 10.1016/j.electacta.2020.137493
# bayes_drt2 is not bundled with pyimpspec and must be installed separately.

from dataclasses import dataclass
from numpy import (
    argsort,
    array,
    asarray,
    log10 as log,
    logspace,
    pi,
    where,
)
from pyimpspec.data import DataSet
from pyimpspec.analysis.utility import (
    _calculate_residuals,
    _calculate_pseudo_chisqr,
)
from pyimpspec.analysis.drt.result import DRTResult
from pyimpspec.exceptions import DRTError
from pyimpspec.progress import Progress
from pyimpspec.typing import (
    ComplexImpedance,
    ComplexImpedances,
    Frequencies,
    Gamma,
    Gammas,
    Indices,
    TimeConstant,
    TimeConstants,
)
from pyimpspec.typing.helpers import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    _is_boolean,
    _is_floating,
    _is_integer,
)


@dataclass(frozen=True)
class BayesDRT2Result(DRTResult):
    """
    An object representing the results of calculating the distribution of relaxation times in a data set using the hierarchical Bayesian inversion implemented by the optional bayes_drt2 package.

    Parameters
    ----------
    time_constants: |TimeConstants|
        The time constants.

    gammas: |Gammas|
        The gamma values (the maximum a posteriori estimate or the posterior mean).

    frequencies: |Frequencies|
        The frequencies of the impedance spectrum.

    impedances: |ComplexImpedances|
        The impedance produced by the model.

    residuals: |ComplexResiduals|
        The residuals of the impedances of the model and the data set.

    mean_gammas: |Gammas|
        The posterior mean gamma values.
        Empty unless credible intervals were requested (i.e., HMC sampling was used).

    lower_bounds: |Gammas|
        The lower bound gamma values of the 95 % credible intervals.

    upper_bounds: |Gammas|
        The upper bound gamma values of the 95 % credible intervals.

    pseudo_chisqr: float
        The pseudo chi-squared value, |pseudo chi-squared|, of the modeled impedance (eq. 14 in Boukamp, 1995).

    series_resistance: float
        The series resistance (ohms) estimated by the model.

    inductance: float
        The inductance (henries) estimated by the model.
    """

    gammas: Gammas
    mean_gammas: Gammas
    lower_bounds: Gammas
    upper_bounds: Gammas
    series_resistance: float
    inductance: float

    def get_label(self) -> str:
        return "Bayesian"

    def get_gammas(self) -> Gammas:
        """
        Get the gamma values.

        Returns
        -------
        |Gammas|
        """
        return self.gammas

    def get_drt_credible_intervals_data(
        self,
    ) -> Tuple[TimeConstants, Gammas, Gammas, Gammas]:
        """
        Get the data necessary to plot the Bayesian credible intervals for this DRT result: the time constants, the mean gamma values, the lower bound gamma values, and the upper bound gamma values.

        Returns
        -------
        Tuple[|TimeConstants|, |Gammas|, |Gammas|, |Gammas|]
        """
        if not self.mean_gammas.size:
            return (
                array([], dtype=TimeConstant),
                array([], dtype=Gamma),
                array([], dtype=Gamma),
                array([], dtype=Gamma),
            )

        return (
            self.time_constants,
            self.mean_gammas,
            self.lower_bounds,
            self.upper_bounds,
        )

    def get_drt_data(self) -> Tuple[TimeConstants, Gammas]:
        """
        Get the data necessary to plot this DRTResult as a DRT plot: the time constants and the corresponding gamma values.

        Returns
        -------
        Tuple[|TimeConstants|, |Gammas|]
        """
        return (
            self.time_constants,  # type: ignore
            self.gammas,  # type: ignore
        )

    def get_peaks(self, threshold: float = 0.0) -> Tuple[TimeConstants, Gammas]:
        """
        Get the time constants (in seconds) and gamma (in ohms) of peaks with magnitudes greater than the threshold.
        The threshold and the magnitudes are all relative to the magnitude of the highest peak.

        Parameters
        ----------
        threshold: float, optional
            The minimum peak height threshold (relative to the height of the tallest peak) for a peak to be included.

        Returns
        -------
        Tuple[|TimeConstants|, |Gammas|]
        """
        indices: Indices = self._get_peak_indices(
            threshold,
            self.gammas,  # type: ignore
        )

        return (
            self.time_constants[indices],  # type: ignore
            self.gammas[indices],  # type: ignore
        )

    def to_peaks_dataframe(
        self,
        threshold: float = 0.0,
        columns: Optional[List[str]] = None,
    ) -> "DataFrame":  # noqa: F821
        from pandas import DataFrame

        if columns is None:
            columns = [
                "tau (s)",
                "gamma (ohm)",
            ]
        elif not isinstance(columns, list):
            raise TypeError(f"Expected a list of strings instead of {columns=}")
        elif len(columns) != 2:
            raise ValueError(f"Expected a list with 2 items instead of {len(columns)=}")
        elif not all(map(lambda s: isinstance(s, str), columns)):
            raise TypeError(f"Expected a list of strings instead of {columns=}")
        elif len(set(columns)) != 2:
            raise ValueError(
                f"Expected a list of 2 unique strings instead of {columns=}"
            )

        time_constants: TimeConstants
        gammas: Gammas
        time_constants, gammas = self.get_peaks(threshold=threshold)

        return DataFrame.from_dict(
            {
                columns[0]: time_constants,  # type: ignore
                columns[1]: gammas,  # type: ignore
            }
        )

    def to_statistics_dataframe(
        self,
    ) -> "DataFrame":  # noqa: F821
        from pandas import DataFrame

        statistics: Dict[str, Union[int, float, str]] = {
            "Log pseudo chi-squared": log(self.pseudo_chisqr),
            "Series resistance (ohm)": self.series_resistance,
            "Inductance (H)": self.inductance,
        }

        return DataFrame.from_dict(
            {
                "Label": list(statistics.keys()),
                "Value": list(statistics.values()),
            }
        )


_MODES: Dict[str, str] = {
    "complex": "both",
    "real": "real",
    "imaginary": "imag",
}

_LOW_FREQUENCY_PRIORS: List[str] = ["free", "negative"]

_LOW_FREQUENCY_MODEL: str = "Series_lowfreqneg.stan"


def _import_inverter():
    try:
        from bayes_drt2.inversion import Inverter
    except ImportError as err:
        raise DRTError(
            "Failed to import bayes_drt2! The Bayesian DRT method requires the optional bayes_drt2 package (and cmdstanpy with CmdStan) to be installed."
            f" ({err})"
        )

    return Inverter


def calculate_drt_bayes_drt2(
    data: DataSet,
    mode: str = "complex",
    credible_intervals: bool = False,
    num_per_decade: int = 10,
    cutoff_frequency: float = 0.0,
    low_frequency_prior: str = "free",
    inductance: bool = True,
    warmup: int = 200,
    samples: int = 200,
    chains: int = 2,
    **kwargs,
) -> BayesDRT2Result:
    """
    Calculates the distribution of relaxation times (DRT) for a given data set using the hierarchical Bayesian inversion implemented by the optional bayes_drt2 package.
    The package (as well as cmdstanpy and CmdStan) must be installed separately.
    The prior on the DRT coefficients is non-negative, except below an optional cut-off frequency where the coefficients may be unconstrained or non-positive.

    References:

    - `Huang, J., Papac, M., and O'Hayre, R., 2020, Electrochim. Acta, 367, 137493 <https://doi.org/10.1016/j.electacta.2020.137493>`_

    Parameters
    ----------
    data: DataSet
        The data set to use in the calculations.

    mode: str, optional
        Which parts of the data are to be included in the calculations.
        Valid values include:

        - "complex"
        - "real"
        - "imaginary"

    credible_intervals: bool, optional
        If true, then Hamiltonian Monte Carlo (HMC) sampling is used to also obtain 95 % credible intervals (slow).
        Otherwise, a maximum a posteriori (MAP) estimate is obtained by optimization (fast).

    num_per_decade: int, optional
        The number of basis functions (and of points of the resulting DRT) per decade of time constants.

    cutoff_frequency: float, optional
        Below this frequency (in hertz), the sign of the DRT is no longer constrained to be non-negative.
        A value of zero means that the DRT is non-negative at all frequencies.

    low_frequency_prior: str, optional
        How the DRT is constrained below the cut-off frequency.
        Valid values include:

        - "free" - the gamma values may be positive or negative
        - "negative" - the gamma values must be non-positive

    inductance: bool, optional
        If true, then an inductance is fitted in addition to the series resistance.

    warmup: int, optional
        The number of warmup iterations of each HMC chain.

    samples: int, optional
        The number of samples drawn by each HMC chain.

    chains: int, optional
        The number of HMC chains.

    Returns
    -------
    |BayesDRT2Result|
    """
    if not isinstance(mode, str):
        raise TypeError(f"Expected a string instead of {mode=}")
    elif mode not in _MODES:
        raise ValueError("Valid mode values: '" + "', '".join(_MODES))

    if not _is_boolean(credible_intervals):
        raise TypeError(f"Expected a boolean instead of {credible_intervals=}")

    if not _is_integer(num_per_decade):
        raise TypeError(f"Expected an integer instead of {num_per_decade=}")
    elif num_per_decade < 1:
        raise ValueError(f"Expected a value greater than zero instead of {num_per_decade=}")

    if not _is_floating(cutoff_frequency):
        raise TypeError(f"Expected a float instead of {cutoff_frequency=}")
    elif cutoff_frequency < 0.0:
        raise ValueError(f"Expected a value greater than or equal to zero instead of {cutoff_frequency=}")

    if not isinstance(low_frequency_prior, str):
        raise TypeError(f"Expected a string instead of {low_frequency_prior=}")
    elif low_frequency_prior not in _LOW_FREQUENCY_PRIORS:
        raise ValueError(
            "Valid low_frequency_prior values: '" + "', '".join(_LOW_FREQUENCY_PRIORS)
        )

    if not _is_boolean(inductance):
        raise TypeError(f"Expected a boolean instead of {inductance=}")

    for name, value in (("warmup", warmup), ("samples", samples), ("chains", chains)):
        if not _is_integer(value):
            raise TypeError(f"Expected an integer instead of {name}={value}")
        elif value < 1:
            raise ValueError(f"Expected a value greater than zero instead of {name}={value}")

    Inverter = _import_inverter()

    f: Frequencies = data.get_frequencies()
    if len(f) < 1:
        raise ValueError(
            f"There are no unmasked data points in the '{data.get_label()}' data set parsed from '{data.get_path()}'"
        )

    Z_exp: ComplexImpedances = data.get_impedances()

    # bayes_drt2 expects the frequencies in descending order.
    order = argsort(f)[::-1]
    f_sorted: Frequencies = f[order]
    Z_sorted: ComplexImpedances = Z_exp[order]

    # Basis functions: one decade of margin on each side of the range of the
    # data, which is the default of bayes_drt2.
    log_tau_min: float = log(1 / (2 * pi * f_sorted.max())) - 1
    log_tau_max: float = log(1 / (2 * pi * f_sorted.min())) + 1
    tau_basis = logspace(
        log_tau_min,
        log_tau_max,
        int(num_per_decade * (log_tau_max - log_tau_min) + 1),
    )

    fit_kwargs: dict = {
        "part": _MODES[mode],
        "nonneg": True,
        "init_from_ridge": True,
    }
    if cutoff_frequency > 0.0:
        tau_cutoff: float = 1 / (2 * pi * cutoff_frequency)
        below = tau_basis >= tau_cutoff  # noqa: F821
        # +/-10 corresponds to unbounded when Z is normalized by bayes_drt2.
        fit_kwargs["model_str"] = _LOW_FREQUENCY_MODEL
        fit_kwargs["add_stan_data"] = {
            "x_lb": where(below, -10.0, 0.0),
            "x_ub": where(below, 0.0 if low_frequency_prior == "negative" else 10.0, 10.0),
        }

    if credible_intervals:
        fit_kwargs.update(
            mode="sample",
            warmup=warmup,
            samples=samples,
            chains=chains,
        )
    else:
        fit_kwargs.update(mode="optimize")

    # The time constants of the resulting DRT.
    tau: TimeConstants = logspace(
        log(1 / f_sorted.max()) - 0.5,
        log(1 / f_sorted.min()) + 0.5,
        int(num_per_decade * (log(f_sorted.max() / f_sorted.min()) + 1.0)) + 1,
    )

    prog: Progress
    with Progress("Fitting model", total=2) as prog:
        try:
            inverter = Inverter(
                basis_freq=1 / (2 * pi * tau_basis),
                fit_inductance=inductance,
            )
            inverter.fit(f_sorted, Z_sorted, **fit_kwargs)
        except Exception as err:
            raise DRTError(f"Failed to perform the Bayesian inversion: {err}")

        prog.increment()
        prog.set_message("Evaluating results")

        gamma: Gammas = asarray(inverter.predict_distribution(tau=tau), dtype=Gamma)
        mean_gamma: Gammas
        lower_gamma: Gammas
        upper_gamma: Gammas
        if credible_intervals:
            mean_gamma = gamma.copy()
            lower_gamma = asarray(
                inverter.predict_distribution(tau=tau, percentile=2.5), dtype=Gamma
            )
            upper_gamma = asarray(
                inverter.predict_distribution(tau=tau, percentile=97.5), dtype=Gamma
            )
        else:
            mean_gamma = array([], dtype=Gamma)
            lower_gamma = array([], dtype=Gamma)
            upper_gamma = array([], dtype=Gamma)

        Z_fit: ComplexImpedances = asarray(
            inverter.predict_Z(f), dtype=ComplexImpedance
        )

    series_resistance: float = float(inverter.R_inf)
    inductance_value: float = float(inverter.inductance)

    return BayesDRT2Result(
        time_constants=tau,
        gammas=gamma,
        frequencies=f,
        impedances=Z_fit,
        residuals=_calculate_residuals(Z_exp, Z_fit),
        mean_gammas=mean_gamma,
        lower_bounds=lower_gamma,
        upper_bounds=upper_gamma,
        pseudo_chisqr=_calculate_pseudo_chisqr(Z_exp, Z_fit),
        series_resistance=series_resistance,
        inductance=inductance_value,
    )
