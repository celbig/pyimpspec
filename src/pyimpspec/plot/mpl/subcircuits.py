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

from pyimpspec.analysis import FitResult
from pyimpspec.analysis.utility import _interpolate
from pyimpspec.circuit.base import (
    Connection,
    Element,
    _calculate_impedances,
)
from pyimpspec.circuit.circuit import Circuit
from pyimpspec.circuit.parallel import Parallel
from pyimpspec.circuit.resistor import Resistor
from pyimpspec.circuit.series import Series
from pyimpspec.plot.colors import (
    COLOR_BLUE,
    COLOR_CYAN,
    COLOR_GRAY,
    COLOR_MAGENTA,
    COLOR_ORANGE,
    COLOR_TEAL,
)
from pyimpspec.typing import (
    ComplexImpedances,
    Frequencies,
)
from pyimpspec.typing.helpers import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    _is_boolean,
)
from numpy import isclose
from .markers import MARKER_CIRCLE
from .helpers import (
    _get_marker_color_args,
    _initialize_figure,
    _validate_figure,
)
from .nyquist import _adjust_nyquist_axes


_DEFAULT_COLORS: List[str] = [
    COLOR_BLUE,
    COLOR_TEAL,
    COLOR_ORANGE,
    COLOR_MAGENTA,
    COLOR_CYAN,
    COLOR_GRAY,
]
_OFFSETS: Tuple[str, ...] = (
    "none",
    "series_resistance",
)


def _get_component_label(
    circuit: Circuit,
    item: Union[Element, Connection],
    identifiers: Dict[Element, int],
) -> str:
    if isinstance(item, Element):
        return circuit.get_element_name(item, identifiers)

    label: str = "".join(
        _get_component_label(circuit, child, identifiers) for child in item
    )
    if isinstance(item, Parallel):
        return f"({label})"
    elif isinstance(item, Series):
        return f"[{label}]"

    raise NotImplementedError(f"Unsupported connection: '{type(item)}'")


def _get_series_components(
    circuit: Circuit,
) -> List[Tuple[str, Union[Element, Connection]]]:
    """
    Get the label and the element/connection of each item in the top-level series connection of a circuit.
    """
    identifiers: Dict[Element, int] = circuit.generate_element_identifiers(running=False)

    return [
        (_get_component_label(circuit, item, identifiers), item) for item in circuit
    ]


def plot_subcircuits(
    data: Union[FitResult, Circuit],
    frequencies: Optional[Frequencies] = None,
    offset: str = "none",
    colors: Optional[List[str]] = None,
    linestyle: str = "--",
    num_per_decade: int = 100,
    figure: Optional["Figure"] = None,  # noqa: F821
    legend: bool = True,
    axes: Optional[List["Axes"]] = None,  # noqa: F821
    adjust_axes: bool = True,
    **kwargs,
) -> Tuple["Figure", List["Axes"]]:  # noqa: F821
    """
    Plot the contribution of each component of a circuit's top-level series connection as a Nyquist plot (-Im(Z) vs Re(Z)).
    For example, the components of ``R(RC)(RW)`` are ``R``, ``(RC)``, and ``(RW)``.
    Since the impedances of elements connected in series add up, the individual contributions sum up to the impedance of the whole circuit.

    Parameters
    ----------
    data: Union[FitResult, Circuit]
        The result of a circuit fit or a circuit.

    frequencies: Optional[|Frequencies|], optional
        The frequencies (in hertz) at which to calculate the contributions.
        Required when ``data`` is a |Circuit|.
        Ignored when ``data`` is a |FitResult|, in which case the frequencies of the fit are used instead.

    offset: str, optional
        Where each contribution is placed on the real axis.
        'none': each contribution starts from the origin and the values are exact.
        'series_resistance': the sum of the resistors in the top-level series connection (e.g., the ohmic resistance) is added to the real part of each of the other contributions and the resistors are not plotted separately.

    colors: Optional[List[str]], optional
        The colors to cycle through when plotting the contributions.

    linestyle: str, optional
        The style of the lines.

    num_per_decade: int, optional
        The number of points per decade used to draw the lines when plotting a |FitResult|.

    figure: Optional[|Figure|], optional
        The matplotlib.figure.Figure instance to use when plotting the data.

    legend: bool, optional
        Whether or not to add a legend.

    axes: Optional[List[|Axes|]], optional
        A list of matplotlib.axes.Axes instances to use when plotting the data.

    adjust_axes: bool, optional
        Whether or not to adjust the axes (label, scale, limits, etc.).

    **kwargs

    Returns
    -------
    Tuple[|Figure|, List[|Axes|]]
    """
    from matplotlib.axes import Axes

    circuit: Circuit
    if isinstance(data, FitResult):
        circuit = data.circuit
        frequencies = _interpolate(
            data.get_frequencies(),
            num_per_decade=num_per_decade,
        )
    elif isinstance(data, Circuit):
        circuit = data
        if frequencies is None:
            raise ValueError(f"Expected frequencies when plotting a Circuit instead of {frequencies=}")
    else:
        raise TypeError(f"Expected a FitResult or a Circuit instead of {data=}")

    if offset not in _OFFSETS:
        raise ValueError(f"Expected one of {_OFFSETS} instead of {offset=}")

    if colors is None:
        colors = _DEFAULT_COLORS
    elif not (isinstance(colors, list) and len(colors) > 0):
        raise TypeError(f"Expected a non-empty list or None instead of {colors=}")

    if not isinstance(linestyle, str):
        raise TypeError(f"Expected a string instead of {linestyle=}")

    if figure is None:
        figure, axes = _initialize_figure(num_rows=1, num_cols=1)
    assert axes is not None

    _validate_figure(figure, axes, num_axes=1)
    axis: Axes = axes[0]

    components: List[Tuple[str, Union[Element, Connection]]] = _get_series_components(circuit)

    series_resistance: float = 0.0
    if offset == "series_resistance":
        series_resistance = sum(
            item.get_values()["R"] for _, item in components if isinstance(item, Resistor)
        )
        components = [(_, item) for _, item in components if not isinstance(item, Resistor)]

    i: int
    for i, (label, item) in enumerate(components):
        Z: ComplexImpedances = _calculate_impedances(item, frequencies)
        x = Z.real + series_resistance
        y = -Z.imag
        color: str = colors[i % len(colors)]

        if isclose(x, x[0]).all() and isclose(y, y[0]).all():
            axis.scatter(
                x,
                y,
                marker=MARKER_CIRCLE,
                **_get_marker_color_args(MARKER_CIRCLE, color),
                label=label,
            )
        else:
            axis.plot(
                x,
                y,
                color=color,
                linestyle=linestyle,
                label=label,
            )

    if not _is_boolean(adjust_axes):
        raise TypeError(f"Expected a boolean instead of {adjust_axes=}")
    elif adjust_axes:
        _adjust_nyquist_axes(axis, admittance=False)

    if not _is_boolean(legend):
        raise TypeError(f"Expected a boolean instead of {legend=}")
    elif legend and len(components) > 0:
        axis.legend()

    return (
        figure,
        axes,
    )
