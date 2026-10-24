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

# This module implements analysis of peaks in DRT spectra by fitting skew
# normal distributions.See 10.3390/batteries5030053 for more information.
# Peaks can also be modeled using Gaussian, ZARC, or Havriliak-Negami
# functions (inspired by pyDRTtools) where the area of each peak is the
# resistance of the corresponding relaxation process.

from abc import (
    ABC,
    abstractmethod,
)
from pyimpspec.progress import Progress
from dataclasses import dataclass
from scipy.signal import (
    find_peaks,
    peak_widths,
)
from numpy import (
    allclose,
    arccosh,
    arctan2,
    argmax,
    argmin,
    argwhere,
    clip,
    copy,
    cos,
    cosh,
    diff,
    exp,
    float64,
    isclose,
    linspace,
    log as ln,
    log10 as log,
    logspace,
    mean,
    pi,
    sign,
    sin,
    sqrt,
    tan,
    trapezoid,
    zeros,
)
from numpy.typing import NDArray
from pyimpspec.typing import (
    Gamma,
    Gammas,
    Indices,
    TimeConstants,
)
from pyimpspec.typing.helpers import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    _is_boolean,
    _is_floating_array,
    _is_floating_list,
    _is_integer,
    _is_integer_array,
    _is_integer_list,
)


def _skew_normal(x, h, p, a, s):
    dp = x - p
    den = 2 * s**2
    num = (dp * (1 + a * sign(dp)))**2

    return h * exp(-num/den)


def _function(
    x,
    parameters: "Parameters",  # noqa: F821
) -> NDArray[float64]:
    params = parameters.valuesdict()
    assert len(params) % 2 == 0, (len(params), params.keys())

    gammas: NDArray[float64] = zeros(len(x), dtype=float64)

    i = 0
    while True:
        try:
            h = params[f"h_{i}"]
        except KeyError:
            break
        p = params[f"p_{i}"]
        alpha = params[f"alpha_{i}"]
        sigma = params[f"sigma_{i}"]

        gammas += _skew_normal(x, h, p, alpha, sigma)
        i += 1

    return gammas


def _residual(
    parameters: "Parameters",  # noqa: F821
    x: NDArray[float64],
    y: NDArray[float64],
):
    return _function(x, parameters) - y


_PEAK_SHAPES: Tuple[str, ...] = (
    "skew-normal",
    "gaussian",
    "zarc",
    "havriliak-negami",
)


class BaseDRTPeak(ABC):
    """
    The base class for peaks in a distribution of relaxation times.
    """

    @abstractmethod
    def get_gammas(
        self,
        time_constants: TimeConstants,
    ) -> Gammas:
        """
        Calculate the gamma values (ohm) corresponding to the given time constants.

        Parameters
        ----------
        time_constants: |TimeConstants|

        Returns
        -------
        |Gammas|
        """
        pass

    @abstractmethod
    def get_area(
        self,
        time_constants: Optional[TimeConstants] = None,
    ) -> float64:
        """
        Calculate the area (ohm) of this peak.

        Parameters
        ----------
        time_constants: Optional[|TimeConstants|], optional
            If provided, then the area is integrated only over the range of these time constants.
            Otherwise, the area is calculated over all time constants (if supported by the peak type).

        Returns
        -------
        float64
        """
        pass

    @abstractmethod
    def get_time_constant(self) -> float64:
        """
        Get the time constant (s) at which the peak reaches its maximum.

        Returns
        -------
        float64
        """
        pass

    @abstractmethod
    def get_height(self) -> float64:
        """
        Get the maximum gamma value (ohm) of the peak.

        Returns
        -------
        float64
        """
        pass


@dataclass(frozen=True)
class DRTPeak(BaseDRTPeak):
    """
    An object that represents a peak in a distribution of relaxation times.
    The peak is modeled using a (skew) normal distribution.

    Parameters
    ----------
    position: float64
        The relative position of the (skew) normal distribution.

    height: float64
        The relative height of the (skew) normal distribution.

    alpha: float64
        The skew of the (skew) normal distribution.

    sigma: float64
        The standard deviation of the (skew) normal distribution.

    x_offset: float64
        The offset used to translate time constants to the relative x-axis.

    x_scale: float64
        The scaling factor used to translate time constants to the relative x-axis.

    y_offset: float64
        The offset used to translate the relative y-axis values to gamma (ohm).

    y_scale: float64
        The scaling factor used to translate the relative y-axis values to gamma (ohm).
    """
    position: float64  # Relative x-axis
    height: float64  # Relative y-axis
    alpha: float64  # Skew
    sigma: float64  # Standard deviation
    x_offset: float64  # For translating from time constant to relative x-axis
    x_scale: float64  # For translating from time constant to relative x-axis
    y_offset: float64  # For translating relative y-axis to gamma (ohm)
    y_scale: float64  # For translating relative y-axis to gamma (ohm)

    def get_gammas(
        self,
        time_constants: TimeConstants,
    ) -> Gammas:
        """
        Calculate the gamma values (ohm) corresponding to the given time constants.

        Parameters
        ----------
        time_constants: |TimeConstants|

        Returns
        -------
        |Gammas|
        """
        x: NDArray[float64] = (log(time_constants) - self.x_offset) / self.x_scale

        return (_skew_normal(
            x=x,
            h=self.height,
            p=self.position,
            a=self.alpha,
            s=self.sigma,
        ) * self.y_scale) + self.y_offset

    def _get_gamma(self, log_tau: float64) -> float64:
        x: float64 = (log_tau - self.x_offset) / self.x_scale

        return (_skew_normal(
            x=x,
            h=self.height,
            p=self.position,
            a=self.alpha,
            s=self.sigma,
        ) * self.y_scale) + self.y_offset

    def get_area(
        self,
        time_constants: Optional[TimeConstants] = None,
    ) -> float64:
        """
        Calculate the area (ohms) of this peak for a given set of time constants.

        Parameters
        ----------
        time_constants: |TimeConstants|
            The time constants that define the range of integration (required for skew normal peaks).

        Returns
        -------
        float64
        """
        from scipy.integrate import quad

        if time_constants is None:
            raise ValueError("Expected time constants to be provided in order to calculate the area of a skew normal peak!")

        return quad(
            func=self._get_gamma,
            a=log(min(time_constants)),
            b=log(max(time_constants)),
        )[0] / log(exp(1))

    def get_time_constant(self) -> float64:
        """
        Get the time constant (s) at which the peak reaches its maximum.

        Returns
        -------
        float64
        """
        return 10**((self.position * self.x_scale) + self.x_offset)

    def get_height(self) -> float64:
        """
        Get the maximum gamma value (ohm) of the peak.

        Returns
        -------
        float64
        """
        return (self.height * self.y_scale) + self.y_offset



def _gaussian(u, R, u0, sigma):
    return R / (sigma * sqrt(2 * pi)) * exp(-((u - u0) ** 2) / (2 * sigma**2))


def _zarc(u, R, u0, n):
    return R / (2 * pi) * sin(n * pi) / (cosh(n * (u - u0)) + cos(n * pi))


def _havriliak_negami(u, R, u0, a, b):
    x = exp(clip(a * (u - u0), -300.0, 300.0))
    theta = arctan2(sin(a * pi), x + cos(a * pi))
    den = (1 + 2 * x * cos(a * pi) + x**2) ** (b / 2)

    return R / pi * x**b * sin(b * theta) / den


class _PhysicalDRTPeak(BaseDRTPeak):
    # Base class for peaks defined in terms of the natural logarithm of the
    # time constant, where the area of the peak is the resistance R.
    R: float64
    tau: float64

    @abstractmethod
    def _get_gamma(self, u):
        pass

    def get_gammas(
        self,
        time_constants: TimeConstants,
    ) -> Gammas:
        return self._get_gamma(ln(time_constants))

    def get_area(
        self,
        time_constants: Optional[TimeConstants] = None,
    ) -> float64:
        from scipy.integrate import quad

        if time_constants is None:
            return self.R

        return quad(
            func=self._get_gamma,
            a=ln(min(time_constants)),
            b=ln(max(time_constants)),
            points=[ln(self.tau)],
        )[0]

    def _find_maximum(self) -> Tuple[float64, float64]:
        from scipy.optimize import minimize_scalar

        u0: float64 = ln(self.tau)
        u = linspace(u0 - 30.0, u0 + 30.0, 6001)
        i: int = int(argmax(self._get_gamma(u)))
        result = minimize_scalar(
            lambda _: -self._get_gamma(_),
            bounds=(u[max(i - 1, 0)], u[min(i + 1, u.size - 1)]),
            method="bounded",
            options=dict(xatol=1e-10),
        )

        return (exp(result.x), -result.fun)

    def get_time_constant(self) -> float64:
        return self._find_maximum()[0]

    def get_height(self) -> float64:
        return self._find_maximum()[1]


@dataclass(frozen=True)
class DRTPeakGaussian(_PhysicalDRTPeak):
    """
    An object that represents a peak in a distribution of relaxation times.
    The peak is modeled using a Gaussian function of the natural logarithm of the time constant.
    The area of the peak is equal to the parameter ``R``.

    Parameters
    ----------
    R: float64
        The area of the peak (ohm).

    tau: float64
        The time constant (s) at the center of the peak.

    sigma: float64
        The standard deviation of the Gaussian function (in units of the natural logarithm of the time constant).
    """
    R: float64
    tau: float64
    sigma: float64

    def _get_gamma(self, u):
        return _gaussian(u, self.R, ln(self.tau), self.sigma)

    def get_time_constant(self) -> float64:
        return self.tau

    def get_height(self) -> float64:
        return self.R / (self.sigma * sqrt(2 * pi))


@dataclass(frozen=True)
class DRTPeakZARC(_PhysicalDRTPeak):
    """
    An object that represents a peak in a distribution of relaxation times.
    The peak corresponds to the DRT of a ZARC element, :math:`Z = R / (1 + (j \\omega \\tau)^n)`.
    The area of the peak is equal to the resistance ``R``.

    Parameters
    ----------
    R: float64
        The resistance (ohm), which is also the area of the peak.

    tau: float64
        The time constant (s).

    n: float64
        The dispersion parameter.
    """
    R: float64
    tau: float64
    n: float64

    def _get_gamma(self, u):
        return _zarc(u, self.R, ln(self.tau), self.n)

    def get_time_constant(self) -> float64:
        return self.tau

    def get_height(self) -> float64:
        return self.R / (2 * pi) * tan(self.n * pi / 2)


@dataclass(frozen=True)
class DRTPeakHavriliakNegami(_PhysicalDRTPeak):
    """
    An object that represents a peak in a distribution of relaxation times.
    The peak corresponds to the DRT of a Havriliak-Negami element, :math:`Z = R / (1 + (j \\omega \\tau)^a)^b`.
    The area of the peak is equal to the resistance ``R``.

    Parameters
    ----------
    R: float64
        The resistance (ohm), which is also the area of the peak.

    tau: float64
        The characteristic time constant (s).
        Note that the maximum of the peak is not located at this time constant unless ``b`` is equal to one.

    a: float64
        The symmetric broadening exponent.

    b: float64
        The asymmetric broadening exponent.
    """
    R: float64
    tau: float64
    a: float64
    b: float64

    def _get_gamma(self, u):
        return _havriliak_negami(u, self.R, ln(self.tau), self.a, self.b)


@dataclass(frozen=True)
class DRTPeaks:
    """
    An object that represents a collection of peaks in a distribution of relaxation times.

    Parameters
    ----------
    time_constants: |TimeConstants|
        The time constants of the DRT.

    peaks: List[|BaseDRTPeak|]
        A list of the peaks fitted to the DRT.

    suffix: str
        The suffix used in the labels when plotting the results.

    shape: str, optional
        The type of function that was fitted to the peaks: ``"skew-normal"``, ``"gaussian"``, ``"zarc"``, or ``"havriliak-negami"``.
    """
    time_constants: TimeConstants
    peaks: List[BaseDRTPeak]
    suffix: str
    shape: str = "skew-normal"

    def __iter__(self):
        return iter(self.peaks)

    def get_num_peaks(self) -> int:
        """
        Get the number of peaks.

        Returns
        -------
        int
        """
        return len(self.peaks)

    def get_time_constants(
        self,
        num_per_decade: int = 100,
    ) -> TimeConstants:
        """
        Get either the original time constants (``num_per_decade < 1``) or an interpolated set of time constants.

        Parameters
        ----------
        num_per_decade: int, optional
            The number of points per decade to use when generating the time constants.

        Returns
        -------
        |TimeConstants|
        """
        if not _is_integer(num_per_decade):
            raise TypeError(f"Expected an integer instead of {num_per_decade=}")

        if num_per_decade < 1:
            return self.time_constants

        log_tau_min = log(min(self.time_constants))
        log_tau_max = log(max(self.time_constants))

        return logspace(
            log_tau_min,
            log_tau_max,
            num=int(round(log_tau_max - log_tau_min) * num_per_decade) + 1,
        )

    def get_gammas(
        self,
        peak_indices: Optional[List[int]] = None,
        num_per_decade: int = 100,
    ) -> Gammas:
        """
        Calculate the gamma values (ohm) for one or more peaks.

        Parameters
        ----------
        peak_indices: Optional[List[int]], optional
            If indices are specified, then only those peaks will be used.

        num_per_decade: int, optional
            The number of points per decade to use when generating the gamma values.

        Returns
        -------
        |Gammas|
        """
        num_peaks: int = self.get_num_peaks()
        if peak_indices is None:
            pass
        elif not (_is_integer_list(peak_indices) or _is_integer_array(peak_indices)):
            raise TypeError(f"Expected an array of integers instead of {peak_indices=}")
        elif not all(map(lambda i: 0 <= i < num_peaks, peak_indices)):
            raise ValueError(f"Expected 0 <= {peak_indices=} < {num_peaks}")

        time_constants: TimeConstants = self.get_time_constants(
            num_per_decade=num_per_decade,
        )

        gammas: Gammas = zeros(time_constants.shape, dtype=Gamma)

        i: int
        peak: BaseDRTPeak
        for i, peak in enumerate(self.peaks):
            if peak_indices and i not in peak_indices:
                continue

            gammas += peak.get_gammas(time_constants=time_constants)

        return gammas

    def get_peak_area(
        self,
        index: int,
    ) -> float64:
        """
        Calculate the area of a peak.

        Parameters
        ----------
        index: int
            The index (zero-based) of the peak for which to calculate the area.

        Returns
        -------
        float64
        """
        from scipy.integrate import quad

        num_peaks: int = self.get_num_peaks()
        if not _is_integer(index):
            raise TypeError(f"Expected an integer instead of {index=}")
        elif not (0 <= index < num_peaks):
            raise ValueError(f"Expected 0 <= {index=} < {num_peaks}")

        peak: BaseDRTPeak = self.peaks[index]
        if isinstance(peak, DRTPeak):
            return quad(
                func=lambda x: peak.get_gammas(10**x),
                a=log(min(self.time_constants)),
                b=log(max(self.time_constants)),
            )[0] / log(exp(1))

        return peak.get_area()

    def to_peaks_dataframe(
        self,
        peak_indices: Optional[List[int]] = None,
    ) -> "DataFrame":  # noqa: F821
        """
        Generate a `pandas.DataFrame`_ object containing information about the peaks.

        Parameters
        ----------
        peak_indices: Optional[List[int]], optional
            If indices are specified, then only those peaks will be used.

        Returns
        -------
        `pandas.DataFrame`_
        """
        from pandas import DataFrame

        num_peaks: int = self.get_num_peaks()
        if peak_indices is None:
            pass
        elif not (_is_integer_list(peak_indices) or _is_integer_array(peak_indices)):
            raise TypeError(f"Expected an array of integers instead of {peak_indices=}")
        elif not all(map(lambda i: 0 <= i < num_peaks, peak_indices)):
            raise ValueError(f"Expected 0 <= {peak_indices=} < {num_peaks}")

        positions: List[float64] = []
        heights: List[float64] = []
        areas: List[float64] = []

        i: int
        peak: BaseDRTPeak
        for i, peak in enumerate(self.peaks):
            if peak_indices and i not in peak_indices:
                continue

            positions.append(peak.get_time_constant())
            heights.append(peak.get_height())
            areas.append(self.get_peak_area(i))

        suffix: str = ""
        if self.suffix != "":
            suffix = f", {self.suffix}"

        return DataFrame.from_dict({
            "tau" + suffix + " (s)": positions,
            "gamma" + suffix + " (ohm)": heights,
            "R_peak" + suffix + " (ohm)": areas,
        })


def _detect_peaks(
    time_constants: TimeConstants,
    x: NDArray[float64],
    y: NDArray[float64],
    num_peaks: int,
    peak_positions: Optional[Union[List[float], NDArray[float64]]],
) -> List[int]:
    # Returns the indices of the peaks to analyze.
    indices: List[int] = []

    if peak_positions is not None and len(peak_positions) > 0:
        pos: float
        for pos in peak_positions:
            indices.append(int(argmin(abs(time_constants - pos))))

        return indices

    y_ext: NDArray[float64] = zeros(len(y) + 2, dtype=float64)
    y_ext[1:-1] += y

    indices = [int(i) - 1 for i in find_peaks(y_ext)[0]]

    if num_peaks > 0:
        indices.sort(key=lambda i: y[i], reverse=True)
        indices = indices[:num_peaks]

    indices.sort(key=lambda i: x[i])

    return indices


def _fit_parameters(
    residual: Callable,
    parameters: "Parameters",  # noqa: F821
    args: tuple,
    global_optimization: bool,
) -> "MinimizerResult":  # noqa: F821
    from lmfit.minimizer import minimize

    if global_optimization:
        # Similar to pyDRTtools: a global search followed by a local refinement.
        from scipy.optimize import differential_evolution

        names: List[str] = [k for k, v in parameters.items() if v.vary]
        if len(names) > 0:
            trial: "Parameters" = parameters.copy()  # noqa: F821

            def objective(values) -> float:
                for name, value in zip(names, values):
                    trial[name].value = value

                return float((residual(trial, *args) ** 2).sum())

            result = differential_evolution(
                objective,
                bounds=[(parameters[k].min, parameters[k].max) for k in names],
                seed=42,
                maxiter=200,
                popsize=15,
                polish=False,
            )

            for name, value in zip(names, result.x):
                parameters[name].value = value

    return minimize(
        residual,
        parameters,
        method="leastsq",
        args=args,
    )


def _generate_parameters(
    peaks: List[Tuple[float64, float64]],
    disallow_skew: bool,
) -> Tuple["Parameters", int]:  # noqa: F821
    from lmfit.parameter import Parameters

    if len(peaks) < 1:
        raise ValueError(f"Expected to have at least one peak to analyze!")

    parameters: Parameters = Parameters()
    num_variables: int = 0

    i: int
    x: float64
    y: float64
    for i, (x, y) in enumerate(peaks):
        assert 0.0 <= x <= 1.0, x
        parameters.add(
            name=f"h_{i}",
            value=y,
            min=0.0,
            max=2.0,
        )

        assert 0.0 <= x <= 1.0
        kw = dict(
            name=f"p_{i}",
            value=x,
            min=x * 0.99,
            max=x * 1.01,
        )
        
        if isclose(kw["min"], 0.0):
            kw["min"] = -1e-10

        if isclose(kw["max"], 1.0):
            kw["max"] = 1.0 + 1e-10

        parameters.add(**kw)

        parameters.add(
            name=f"alpha_{i}",
            value=0.0,
            min=-1.0,
            max=1.0,
            vary=not disallow_skew,
        )

        parameters.add(
            name=f"sigma_{i}",
            value=0.05,
            min=1e-10,
            max=1e2,
        )

        num_variables += 3 if disallow_skew else 4

    return (parameters, num_variables)


def _analyze_peaks(
    time_constants: TimeConstants,
    gammas: Gammas,
    num_peaks: int,
    peak_positions: Optional[Union[List[float], NDArray[float64]]],
    disallow_skew: bool = False,
    suffix: str = "",
    shape: str = "skew-normal",
    global_optimization: bool = False,
) -> DRTPeaks:
    from lmfit.minimizer import MinimizerResult
    from lmfit.parameter import Parameters

    if not _is_floating_array(time_constants):
        raise TypeError(f"Expected an array of floats instead of {time_constants=}")
    elif not _is_floating_array(gammas):
        raise TypeError(f"Expected an array of floats instead of {gammas=}")
    elif not _is_integer(num_peaks):
        raise TypeError(f"Expected an integer instead of {num_peaks=}")
    elif not (
        peak_positions is None
        or _is_floating_array(peak_positions)
        or _is_floating_list(peak_positions)
    ):
        raise TypeError(f"Expected either None or a list/array of floats instead of {peak_positions=}")
    elif not _is_boolean(disallow_skew):
        raise TypeError(f"Expected a boolean instead of {disallow_skew=}")
    elif not isinstance(suffix, str):
        raise TypeError(f"Expected a string instead of {suffix=}")
    elif not isinstance(shape, str):
        raise TypeError(f"Expected a string instead of {shape=}")
    elif shape not in _PEAK_SHAPES:
        raise ValueError(f"Expected one of {_PEAK_SHAPES} instead of {shape=}")
    elif not _is_boolean(global_optimization):
        raise TypeError(f"Expected a boolean instead of {global_optimization=}")

    if allclose(gammas, 0.0):
        return DRTPeaks(
            time_constants=time_constants,
            peaks=[],
            suffix=suffix,
            shape=shape,
        )

    if shape != "skew-normal":
        return _analyze_peaks_physical(
            time_constants=time_constants,
            gammas=gammas,
            num_peaks=num_peaks,
            peak_positions=peak_positions,
            suffix=suffix,
            shape=shape,
            global_optimization=global_optimization,
        )

    prog: Progress
    with Progress(
        "Scaling data",
        total=4,
    ) as prog:
        x: NDArray[float64] = log(time_constants)
        x_offset: float64 = min(x)
        x -= x_offset
        x_scale: float64 = max(x)
        x /= x_scale
        assert isclose(min(x), 0.0)
        assert isclose(max(x), 1.0)
        prog.increment()

        y: NDArray[float64] = copy(gammas)
        indices: Indices = argwhere(y < 0.0).flatten()
        y[indices] = 0.0
        y_offset: float64 = min(y)
        y = y - y_offset
        y_scale: float64 = max(y)
        y /= y_scale
        assert isclose(min(y), 0.0), min(y)
        assert isclose(max(y), 1.0), max(y)
        prog.increment()

        prog.set_message("Detecting peaks")
        i: int
        peaks: List[Tuple[float64, float64]] = [
            (x[i], y[i])
            for i in _detect_peaks(
                time_constants,
                x,
                y,
                num_peaks,
                peak_positions,
            )
        ]

        parameters: Parameters
        num_variables: int
        parameters, num_variables = _generate_parameters(
            peaks,
            disallow_skew=disallow_skew,
        )
        if num_peaks < 1 and peak_positions is None:
            while len(peaks) > 1 and num_variables > x.size:
                peaks.sort(key=lambda t: t[1], reverse=True)
                peaks.pop()
                peaks.sort(key=lambda t: t[0])
                parameters, num_variables = _generate_parameters(
                    peaks,
                    disallow_skew=disallow_skew,
                )

        prog.increment()

        prog.set_message(f"Fitting {len(peaks)} peaks")
        fit: MinimizerResult = _fit_parameters(
            _residual,
            parameters,
            args=(x, y),
            global_optimization=global_optimization,
        )
        params: Dict[str, float64] = fit.params.valuesdict()

        drt_peaks: List[DRTPeak] = []

        for i in range(0, len(peaks)):
            drt_peaks.append(DRTPeak(
                position=params[f"p_{i}"],
                height=params[f"h_{i}"],
                alpha=params[f"alpha_{i}"],
                sigma=params[f"sigma_{i}"],
                x_offset=x_offset,
                x_scale=x_scale,
                y_offset=y_offset,
                y_scale=y_scale,
            ))

    return DRTPeaks(
        time_constants=time_constants,
        peaks=drt_peaks,
        suffix=suffix,
        shape=shape,
    )


def _zarc_fwhm_to_n(fwhm: float64) -> float64:
    from scipy.optimize import brentq

    def f(n: float) -> float:
        return 2 * arccosh(2 + cos(n * pi)) / n - fwhm

    lo: float = 0.05
    hi: float = 0.999
    if f(lo) * f(hi) > 0.0:
        return 0.8

    return brentq(f, lo, hi)


def _generate_parameters_physical(
    shape: str,
    indices: List[int],
    u: NDArray[float64],
    y: NDArray[float64],
) -> "Parameters":  # noqa: F821
    from lmfit.parameter import Parameters

    if len(indices) < 1:
        raise ValueError("Expected to have at least one peak to analyze!")

    du: float64 = mean(diff(u))
    total_area: float64 = trapezoid(y, u)

    y_ext: NDArray[float64] = zeros(len(y) + 2, dtype=float64)
    y_ext[1:-1] = y

    parameters: Parameters = Parameters()

    k: int
    i: int
    for k, i in enumerate(indices):
        widths = peak_widths(y_ext, [i + 1], rel_height=0.5)[0]
        fwhm: float64 = max(float(widths[0]) * du, 3 * du)
        height: float64 = y[i]

        parameters.add(
            name=f"R_{k}",
            value=min(height * fwhm * 1.0645, 1.5 * total_area),
            min=0.0,
            max=2.0 * total_area,
        )
        parameters.add(
            name=f"u_{k}",
            value=u[i],
            min=max(u[i] - fwhm, u[0]),
            max=min(u[i] + fwhm, u[-1]),
        )

        if shape == "gaussian":
            parameters.add(
                name=f"sigma_{k}",
                value=min(max(fwhm / 2.355, du / 2), u[-1] - u[0]),
                min=du / 2,
                max=u[-1] - u[0],
            )
        else:
            n: float64 = _zarc_fwhm_to_n(fwhm)
            if shape == "zarc":
                parameters.add(name=f"n_{k}", value=n, min=0.05, max=0.999)
            else:
                parameters.add(name=f"a_{k}", value=n, min=0.05, max=0.999)
                parameters.add(name=f"b_{k}", value=1.0, min=0.05, max=1.0)

    return parameters


def _function_physical(
    shape: str,
    u: NDArray[float64],
    params: Dict[str, float64],
) -> NDArray[float64]:
    gammas: NDArray[float64] = zeros(len(u), dtype=float64)

    k: int = 0
    while f"R_{k}" in params:
        R: float64 = params[f"R_{k}"]
        u0: float64 = params[f"u_{k}"]
        if shape == "gaussian":
            gammas += _gaussian(u, R, u0, params[f"sigma_{k}"])
        elif shape == "zarc":
            gammas += _zarc(u, R, u0, params[f"n_{k}"])
        else:
            gammas += _havriliak_negami(
                u,
                R,
                u0,
                params[f"a_{k}"],
                params[f"b_{k}"],
            )
        k += 1

    return gammas


def _residual_physical(
    parameters: "Parameters",  # noqa: F821
    shape: str,
    u: NDArray[float64],
    y: NDArray[float64],
) -> NDArray[float64]:
    return _function_physical(shape, u, parameters.valuesdict()) - y


def _analyze_peaks_physical(
    time_constants: TimeConstants,
    gammas: Gammas,
    num_peaks: int,
    peak_positions: Optional[Union[List[float], NDArray[float64]]],
    suffix: str,
    shape: str,
    global_optimization: bool,
) -> DRTPeaks:
    from lmfit.minimizer import MinimizerResult

    num_params: int = {"gaussian": 3, "zarc": 3, "havriliak-negami": 4}[shape]

    prog: Progress
    with Progress(
        "Scaling data",
        total=4,
    ) as prog:
        u: NDArray[float64] = ln(time_constants)
        y: NDArray[float64] = copy(gammas)
        y[y < 0.0] = 0.0
        y_scale: float64 = max(y)
        if y_scale <= 0.0:
            return DRTPeaks(
                time_constants=time_constants,
                peaks=[],
                suffix=suffix,
                shape=shape,
            )
        y /= y_scale
        prog.increment()

        prog.set_message("Detecting peaks")
        indices: List[int] = _detect_peaks(
            time_constants,
            u,
            y,
            num_peaks,
            peak_positions,
        )
        if num_peaks < 1 and peak_positions is None:
            while len(indices) > 1 and len(indices) * num_params > u.size:
                indices.remove(min(indices, key=lambda i: y[i]))

        if len(indices) < 1:
            return DRTPeaks(
                time_constants=time_constants,
                peaks=[],
                suffix=suffix,
                shape=shape,
            )

        parameters: Parameters = _generate_parameters_physical(
            shape,
            indices,
            u,
            y,
        )
        prog.increment()

        prog.set_message(f"Fitting {len(indices)} peaks")
        fit: MinimizerResult = _fit_parameters(
            _residual_physical,
            parameters,
            args=(shape, u, y),
            global_optimization=global_optimization,
        )
        params: Dict[str, float64] = fit.params.valuesdict()
        prog.increment()

    drt_peaks: List[BaseDRTPeak] = []

    k: int
    for k in range(0, len(indices)):
        R: float64 = params[f"R_{k}"] * y_scale
        tau: float64 = exp(params[f"u_{k}"])
        if shape == "gaussian":
            drt_peaks.append(DRTPeakGaussian(R, tau, params[f"sigma_{k}"]))
        elif shape == "zarc":
            drt_peaks.append(DRTPeakZARC(R, tau, params[f"n_{k}"]))
        else:
            drt_peaks.append(
                DRTPeakHavriliakNegami(
                    R,
                    tau,
                    params[f"a_{k}"],
                    params[f"b_{k}"],
                )
            )

    drt_peaks.sort(key=lambda peak: peak.tau)

    return DRTPeaks(
        time_constants=time_constants,
        peaks=drt_peaks,
        suffix=suffix,
        shape=shape,
    )
