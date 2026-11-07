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

from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from .base import (
    Connection,
    Element,
    _calculate_impedances,
)
from .series import Series
from .parallel import Parallel
from .resistor import Resistor
from numpy import (
    argmax,
    array,
    float64,
    inf,
    isclose,
    isfinite,
    logspace,
)
from numpy.typing import NDArray
from sympy import (
    Expr,
    latex,
)
from pyimpspec.typing import (
    ComplexImpedances,
    Frequencies,
)


# CDC syntax version
VERSION: int = 1


ElementParameters = Dict[str, Dict[str, str]]


_SUBCIRCUIT_OFFSETS: Tuple[str, ...] = (
    "none",
    "series_resistance",
    "cumulative_dc",
    "peak",
)


def _get_characteristic_frequency(
    frequencies: Frequencies,
    impedances: ComplexImpedances,
) -> float:
    # The frequency where the magnitude of the imaginary part is the greatest:
    # the apex of an arc, the low-frequency end of a diffusion tail, or the
    # high-frequency end of an inductive tail.
    # Frequency-independent contributions (e.g., resistors) are treated as the fastest.
    imag: NDArray[float64] = abs(impedances.imag)
    if not (imag.max() > 0.0):
        return inf

    return float(frequencies[argmax(imag)])


def _get_dc_resistance(
    item: Union[Element, Connection],
    frequencies: Frequencies,
) -> float:
    # The real part of the impedance at the zero-frequency limit.
    # Contributions without a finite limit (e.g., a semi-infinite Warburg element or a constant phase element
    # connected in series) fall back to the real part at the lowest frequency.
    # The real part is evaluated at decreasingly low frequencies until two consecutive decades agree.
    # The search is limited since the impedances of some elements lose precision at extremely low frequencies.
    f_min: float = float(min(frequencies))
    f_low: Frequencies = f_min * logspace(-1, -14, 14)
    R_low: NDArray[float64] = item.get_impedances(f_low).real
    i: int
    for i in range(1, len(R_low)):
        if not (isfinite(R_low[i - 1]) and isfinite(R_low[i])):
            break
        elif isclose(R_low[i - 1], R_low[i], rtol=1e-3, atol=0.0):
            return float(R_low[i])

    return float(item.get_impedances(array([f_min], dtype=float64))[0].real)


class Circuit:
    """
    A class that represents an equivalent circuit.

    Parameters
    ----------
    elements: Union[List[Element], Element, Series, Parallel]
        The elements of the circuit.
    """

    def __init__(self, elements: Connection):
        if isinstance(elements, Series):
            pass
        elif isinstance(elements, Parallel):
            elements = Series([elements])
        elif isinstance(elements, Element):
            elements = Series([elements])
        elif isinstance(elements, list):
            if not all(map(lambda e: isinstance(e, Element), elements)):
                raise TypeError(f"Expected a List[Element] instead of {elements=}")
            else:
                elements = Series([elements])
        else:
            raise TypeError(
                f"Expected a List[Element], Element, Series, or Parallel instead of {elements=}"
            )

        self._elements: Series = elements

    def __copy__(self) -> "Circuit":
        return type(self)(
            self._elements.__copy__(),  # type: ignore
        )

    def __deepcopy__(self, memo: dict) -> "Circuit":
        ident: int = id(self)
        copy: Optional["Circuit"] = memo.get(ident)

        if copy is None:
            copy = type(self)(
                self._elements.__deepcopy__(memo),  # type: ignore
            )
            memo[ident] = copy

        return copy

    def __iter__(self) -> List[Union[Element, Connection]]:
        return self._elements.__iter__()

    def __repr__(self) -> str:
        return f"Circuit ('{self.to_string()}', {hex(id(self))})"

    def __str__(self) -> str:
        return self.to_string()

    def __contains__(self, element_or_connection: Union[Element, Connection]) -> bool:
        return self._elements.contains(element_or_connection, top_level=False)

    def to_stack(self) -> List[Tuple[str, Union[Element, Connection]]]:
        stack: List[Tuple[str, Union[Element, Connection]]] = []
        self._elements.to_stack(stack)

        return stack

    def serialize(self, decimals: int = 12) -> str:
        """
        Generate a circuit description code (CDC) that represents this circuit.
        This CDC is modified to also include the current version of the CDC syntax supported by pyimpspec.

        Parameters
        ----------
        decimals: int, optional
            The number of decimals to include for the current element parameter values and limits.

        Returns
        -------
        str
        """
        if decimals < 1:
            raise ValueError(f"Expected a value greater than or equal to 1 instead of {decimals=}")

        return (
            "!"
            + "/".join(
                [
                    f"V={VERSION}",  # CDC syntax version
                ]
            )
            + "!"
            + self.to_string(decimals=decimals)
        )

    def to_string(self, decimals: int = -1) -> str:
        """
        Generate the circuit description code (CDC) that represents this circuit.

        Parameters
        ----------
        decimals: int, optional
            The number of decimals to include for the current element parameter values and limits.
            -1 means that the CDC is generated using the basic syntax, which omits element labels, parameter values, and parameter limits.

        Returns
        -------
        str
        """
        return self._elements.to_string(decimals=decimals)

    def get_impedances(self, frequencies: Frequencies) -> ComplexImpedances:
        """
        Calculate the impedance of this circuit at multiple frequencies.

        Parameters
        ----------
        frequencies: |Frequencies|
            The excitation frequencies in hertz.

        Returns
        -------
        |ComplexImpedances|
        """
        return _calculate_impedances(self._elements, frequencies)

    def get_connections(self, recursive: bool = True) -> List[Connection]:
        """
        Get the connections in this circuit.

        Parameters
        ----------
        recursive: bool, optional
            If True and this Circuit contains nested Connection instances, then all of them are returned.
            If False, then only the top-level Connection is returned.

        Returns
        -------
        List[Connection]
        """
        if recursive:
            connections: List[Connection] = self._elements.get_connections(
                recursive=recursive
            )
            connections.insert(0, self._elements)

            return connections

        return [self._elements]

    def get_elements(self, recursive: bool = True) -> List[Element]:
        """
        Get the elements in this circuit.

        Parameters
        ----------
        recursive: bool, optional
            If True and there are Element instances nested within Connection instances, then all Element instances are returned.
            If False, then only the Element instances within the top-level Connection are returned.

        Returns
        -------
        List[Element]
        """
        if recursive:
            return self._elements.get_elements(recursive=recursive)

        return [item for item in self._elements if isinstance(item, Element)]

    def generate_element_identifiers(self, running: bool) -> Dict[Element, int]:
        """
        Generate a mapping of elements to their corresponding integer identifiers.

        Parameters
        ----------
        running: bool
            If true, then the identifiers are simply a running count from 0 to N. Primarily intended for use within pyimpspec.
            If false, then the identifiers represent what number instance of a particular element type an element is (e.g., the second resistor of three resistors would have 2 as its identifier). Primarily intended for use in anything that most users would see (e.g., circuit diagrams and parameter tables).

        Returns
        -------
        Dict[Element, int]
        """
        return self._elements.generate_element_identifiers(running=running)

    def get_element_name(
        self,
        element: Element,
        identifiers: Optional[Dict[Element, int]] = None,
    ) -> str:
        """
        Get the name of the element with consideration for any overriding label assigned to the element or the type-specific count in the context of this circuit.

        Parameters
        ----------
        element: Element
            The element whose name should be returned.

        identifiers: Optional[Dict[Element, int]], optional
            The identifiers to use when determining the name of the provided element.

        Returns
        -------
        str
        """
        return self._elements.get_element_name(element=element, identifiers=identifiers)

    def get_subcircuit_impedances(
        self,
        frequencies: Frequencies,
        offset: str = "none",
        sort: bool = False,
    ) -> List[Tuple[str, ComplexImpedances]]:
        """
        Get the label and the impedances of each component of the top-level series connection of this circuit.
        For example, the components of ``R(RC)(RW)`` are ``R``, ``(RC)``, and ``(RW)``.
        Since the impedances of elements connected in series add up, the individual contributions sum up to the impedance of the whole circuit.

        Parameters
        ----------
        frequencies: |Frequencies|
            The frequencies (in hertz) at which to calculate the impedances.

        offset: str, optional
            Where each contribution is placed on the real axis.
            The offsets are constant (i.e., the shapes of the contributions are preserved) and only affect the real part.
            With the exception of 'none', the offset contributions no longer sum up to the impedance of the circuit and the resistors in the top-level series connection are not included in the returned list.

            - 'none': the contributions are exact and sum up to the impedance of the circuit.
            - 'series_resistance': the sum of the resistors in the top-level series connection (e.g., the ohmic resistance) is added to the real part of each of the other contributions.
            - 'cumulative_dc': the sum of the resistors in the top-level series connection and the zero-frequency resistances of the faster components (i.e., those with higher characteristic frequencies) is added to the real part of each contribution. Each contribution thus starts where it would start if the time constants were well separated. The real part at the lowest frequency is used for components without a finite zero-frequency limit (e.g., a semi-infinite Warburg element).
            - 'peak': the real parts of all of the other components (including resistors) at the characteristic frequency of a contribution are added to the real part of that contribution. The apex of each arc is thus placed at the same real value as the impedance of the whole circuit at that frequency, even when the arcs overlap.

            The characteristic frequency of a component is the frequency where the magnitude of the imaginary part of its impedance is the greatest (e.g., the apex of an arc).

        sort: bool, optional
            Sort the components from the fastest to the slowest (i.e., by decreasing characteristic frequency) instead of keeping the order of the circuit.
            Frequency-independent components (e.g., resistors) are placed first.

        Returns
        -------
        List[Tuple[str, |ComplexImpedances|]]
        """
        if offset not in _SUBCIRCUIT_OFFSETS:
            raise ValueError(f"Expected one of {_SUBCIRCUIT_OFFSETS} instead of {offset=}")

        identifiers: Dict[Element, int] = self.generate_element_identifiers(running=False)
        labels: List[str] = []
        items: List[Union[Element, Connection]] = []
        impedances: List[ComplexImpedances] = []
        for item in self:
            labels.append(self._get_component_label(item, identifiers))
            items.append(item)
            impedances.append(item.get_impedances(frequencies))

        characteristic_frequencies: List[float] = [
            _get_characteristic_frequency(frequencies, Z) for Z in impedances
        ]
        is_resistor: List[bool] = [isinstance(item, Resistor) for item in items]

        offsets: List[float] = [0.0] * len(items)
        if offset != "none":
            series_resistance: float = sum(
                item.get_values()["R"]
                for item, resistor in zip(items, is_resistor)
                if resistor
            )

            i: int
            j: int
            for i in range(len(items)):
                if is_resistor[i]:
                    continue

                if offset == "series_resistance":
                    offsets[i] = series_resistance
                elif offset == "cumulative_dc":
                    offsets[i] = series_resistance + sum(
                        _get_dc_resistance(items[j], frequencies)
                        for j in range(len(items))
                        if not is_resistor[j]
                        and j != i
                        and (
                            characteristic_frequencies[j] > characteristic_frequencies[i]
                            or (
                                characteristic_frequencies[j] == characteristic_frequencies[i]
                                and j < i
                            )
                        )
                    )
                elif offset == "peak":
                    f_k: float = characteristic_frequencies[i]
                    if not isfinite(f_k):
                        f_k = float(min(frequencies))

                    f_peak: Frequencies = array([f_k], dtype=float64)
                    offsets[i] = sum(
                        float(items[j].get_impedances(f_peak)[0].real)
                        for j in range(len(items))
                        if j != i
                    )

        indices: List[int] = list(range(len(items)))
        if sort:
            # Python's sort is stable so the order of the circuit is kept for ties
            indices.sort(key=lambda i: -characteristic_frequencies[i])

        return [
            (labels[i], impedances[i] + offsets[i])
            for i in indices
            if offset == "none" or not is_resistor[i]
        ]

    def _get_component_label(
        self,
        item: Union[Element, Connection],
        identifiers: Dict[Element, int],
    ) -> str:
        if isinstance(item, Element):
            return self.get_element_name(item, identifiers)

        label: str = "".join(
            self._get_component_label(child, identifiers) for child in item
        )
        if isinstance(item, Parallel):
            return f"({label})"
        elif isinstance(item, Series):
            return f"[{label}]"

        raise NotImplementedError(f"Unsupported connection: '{type(item)}'")

    def to_sympy(self, substitute: bool = False) -> Expr:
        """
        Get the |Expr| object corresponding to this circuit's impedance.

        Parameters
        ----------
        substitute: bool, optional
            Whether or not the variables should be substituted with the current values.

        Returns
        -------
        |Expr|
        """
        expr: Expr = self._elements.to_sympy(
            substitute=substitute,
            identifiers=self.generate_element_identifiers(running=True),
        )
        if not isinstance(expr, Expr):
            raise TypeError(f"Expected an Expr instead of {expr=}")

        return expr

    def to_latex(self) -> str:
        """
        Get the LaTeX math expression corresponding to this circuit's impedance.

        Returns
        -------
        str
        """
        return f"Z = {latex(self.to_sympy(substitute=False))}"

    def to_drawing(self) -> "Drawing":  # noqa: F821
        # Dynamically set to pyimpspec.circuit.diagrams.to_drawing
        raise NotImplementedError()

    def to_circuitikz(self) -> str:
        # Dynamically set to pyimpspec.circuit.diagrams.to_circuitikz
        raise NotImplementedError()
