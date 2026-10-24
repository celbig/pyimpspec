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

from dataclasses import (
    dataclass,
    field,
)
from pathlib import Path
from re import (
    Match,
    Pattern,
    compile as compile_regex,
)
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
from warnings import warn
from numpy import (
    errstate,
    float64,
    isfinite,
    sqrt,
)
from pyimpspec.circuit.base import (
    Connection,
    Element,
)
from pyimpspec.circuit.circuit import Circuit
from pyimpspec.circuit.elements import (
    Capacitor,
    ConstantPhaseElement,
    DeLevieFiniteLength,
    Gerischer,
    Inductor,
    Resistor,
    WarburgOpen,
    WarburgShort,
)
from pyimpspec.circuit.parallel import Parallel
from pyimpspec.circuit.series import Series
from pyimpspec.exceptions import (
    ConnectionWithoutElements,
    InsufficientElementsInParallelConnection,
    ParsingError,
)


# ZView element types that only affect the topology of the circuit.
_BEGIN_PARALLEL: int = -1
_END_PARALLEL: int = -2
_BEGIN_SERIES: int = -3
_END_SERIES: int = -4

# Maps the values of the 'Free' fields to whether or not a parameter is fixed.
_FREE_CODES: Dict[int, bool] = {
    0: True,
    1: False,
}

# ZView element types that have no equivalent in pyimpspec.
_UNSUPPORTED_ELEMENTS: Dict[int, str] = {
    92: "theta",
    93: "x(theta) multiplier",
    94: "x(theta) multiplier",
    95: "element follower",
    96: "high-frequency range element",
    97: "low-frequency range element",
    100: "distributed element (DE)",
    200: "extended distributed element (DX)",
}

_BEGIN_MODEL: Pattern = compile_regex(r"^\s*Begin Circuit Model:\s*(\d+)\s*$")
_END_MODEL: Pattern = compile_regex(r"^\s*End Circuit Model:\s*(\d+)\s*$")
_ELEMENT_FIELD: Pattern = compile_regex(r"^\s*Element #(\d+)\s+(Type|Name):\s*(.*?)\s*$")
_PARAMETER_FIELD: Pattern = compile_regex(
    r"^\s*Element #(\d+),(\d+)\s+(Name|Value|Free):\s*(.*?)\s*$"
)


@dataclass
class _MdlParameter:
    name: str = ""
    value: Optional[float] = None
    fixed: Optional[bool] = None


@dataclass
class _MdlElement:
    index: int
    type: Optional[int] = None
    name: str = ""
    parameters: Dict[int, _MdlParameter] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"element #{self.index} ({self.name}, type {self.type})"

    def get(self, index: int) -> Tuple[float64, bool]:
        parameter: Optional[_MdlParameter] = self.parameters.get(index)
        if parameter is None or parameter.value is None or parameter.fixed is None:
            raise ParsingError(f"Expected parameter #{index} to be fully defined for {self}")

        return float64(parameter.value), parameter.fixed


def _read_lines(path: Path) -> List[str]:
    data: bytes = path.read_bytes()
    try:
        return data.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        # ZView is a Windows application, which is why files may use the cp1252 encoding.
        return data.decode("cp1252").splitlines()


def _parse_float(value: str, element: _MdlElement) -> float:
    try:
        # The decimal separator depends on the locale that was used by ZView.
        return float(value.replace(",", "."))
    except ValueError:
        raise ParsingError(f"Expected a numeric value instead of {value=} for {element}")


def _parse_free(value: str, element: _MdlElement) -> bool:
    try:
        code: int = int(value)
    except ValueError:
        raise ParsingError(f"Expected an integer instead of {value=} for {element}")

    if code not in _FREE_CODES:
        warn(f"Unknown 'Free' code {code} for {element}! Treating the parameter as free.")
        return False

    return _FREE_CODES[code]


def _tokenize(lines: List[str]) -> List[_MdlElement]:
    num_elements: Optional[int] = None
    elements: Dict[int, _MdlElement] = {}

    line: str
    for line in lines:
        match: Optional[Match]
        if num_elements is None:
            match = _BEGIN_MODEL.match(line)
            if match is not None:
                num_elements = int(match.group(1))
            continue

        if _END_MODEL.match(line) is not None:
            break

        match = _ELEMENT_FIELD.match(line)
        if match is not None:
            i: int = int(match.group(1))
            element: _MdlElement = elements.setdefault(i, _MdlElement(index=i))
            if match.group(2) == "Type":
                try:
                    element.type = int(match.group(3))
                except ValueError:
                    raise ParsingError(f"Expected an integer type for {element}")
            else:
                element.name = match.group(3)
            continue

        match = _PARAMETER_FIELD.match(line)
        if match is not None:
            i = int(match.group(1))
            element = elements.setdefault(i, _MdlElement(index=i))
            parameter: _MdlParameter = element.parameters.setdefault(
                int(match.group(2)),
                _MdlParameter(),
            )
            key: str = match.group(3)
            value: str = match.group(4)
            if key == "Name":
                parameter.name = value
            elif key == "Value":
                parameter.value = _parse_float(value, element)
            else:
                parameter.fixed = _parse_free(value, element)
            continue

        if line.strip() != "":
            raise ParsingError(f"Unexpected line in circuit model: {line!r}")
    else:
        if num_elements is None:
            raise ParsingError("Expected to find the beginning of a ZView circuit model!")
        raise ParsingError("Expected to find the end of the ZView circuit model!")

    if sorted(elements.keys()) != list(range(num_elements)):
        raise ParsingError(
            f"Expected {num_elements} elements numbered from 0 instead of {sorted(elements.keys())}"
        )

    for element in elements.values():
        if element.type is None:
            raise ParsingError(f"Expected a type to be defined for {element}")

    return [elements[i] for i in range(num_elements)]


def _create(
    Class: Type[Element],
    mdl: _MdlElement,
    parameters: Dict[str, Tuple[float64, List[bool]]],
    used: List[bool],
) -> Element:
    """
    Create an element.

    Parameters
    ----------
    Class: Type[Element]
        The pyimpspec element to create.

    mdl: _MdlElement
        The ZView element that is being converted.

    parameters: Dict[str, Tuple[float64, List[bool]]]
        The pyimpspec parameters, their values, and whether or not each of the ZView parameters that the value depends on is fixed.
        A parameter that does not depend on any ZView parameters is a constant and is fixed.

    used: List[bool]
        Whether or not each ZView parameter that is used by the conversion is fixed.
    """
    element: Element = Class()
    values: Dict[str, float] = {}
    fixed: Dict[str, bool] = {}

    key: str
    value: float64
    dependencies: List[bool]
    for key, (value, dependencies) in parameters.items():
        fixed[key] = all(dependencies)
        if isfinite(value):
            values[key] = float(value)
        else:
            warn(
                f"Unable to convert the value of parameter '{key}' for {mdl}! Using the default value instead."
            )

    element.set_values(**values)
    element.set_fixed(**fixed)

    # Coupled parameters can only be fixed if all of their dependencies are fixed,
    # which may result in more free parameters than in ZView.
    num_free_zview: int = used.count(False)
    num_free: int = list(fixed.values()).count(False)
    if num_free != num_free_zview:
        warn(
            f"The converted {mdl} has {num_free} free parameters instead of {num_free_zview}!"
        )

    try:
        element.set_label(mdl.name)
    except ValueError:
        warn(f"Unable to use the name of {mdl} as a label!")

    return element


def _resistor(mdl: _MdlElement) -> Element:
    R, fR = mdl.get(0)
    return _create(Resistor, mdl, {"R": (R, [fR])}, [fR])


def _capacitor(mdl: _MdlElement) -> Element:
    C, fC = mdl.get(1)
    return _create(Capacitor, mdl, {"C": (C, [fC])}, [fC])


def _inductor(mdl: _MdlElement) -> Element:
    L, fL = mdl.get(1)
    return _create(Inductor, mdl, {"L": (L, [fL])}, [fL])


def _cpe(mdl: _MdlElement) -> Element:
    # Z = 1 / (T*(I*w)^P)
    T, fT = mdl.get(1)
    P, fP = mdl.get(2)
    return _create(
        ConstantPhaseElement,
        mdl,
        {"Y": (T, [fT]), "n": (P, [fP])},
        [fT, fP],
    )


def _qpe(mdl: _MdlElement) -> Element:
    # Z = 1 / (I*w*Q)^n
    Q, fQ = mdl.get(1)
    n, fn = mdl.get(2)
    return _create(
        ConstantPhaseElement,
        mdl,
        {"Y": (Q**n, [fQ, fn]), "n": (n, [fn])},
        [fQ, fn],
    )


def _gerischer(mdl: _MdlElement) -> Element:
    # Z = 1 / (Y0*sqrt(Ka + I*w))
    Y0, fY0 = mdl.get(1)
    Ka, fKa = mdl.get(2)
    return _create(
        Gerischer,
        mdl,
        {"Y": (Y0, [fY0]), "k": (Ka, [fKa]), "n": (float64(0.5), [])},
        [fY0, fKa],
    )


def _warburg(Class: Type[Element]) -> Callable[[_MdlElement], Element]:
    # Z = R*tanh((I*T*w)^P) / (I*T*w)^P (short)
    # Z = R*coth((I*T*w)^P) / (I*T*w)^P (open)
    def wrapper(mdl: _MdlElement) -> Element:
        R, fR = mdl.get(0)
        T, fT = mdl.get(1)
        P, fP = mdl.get(2)
        return _create(
            Class,
            mdl,
            {
                "Y": (T * R ** (-1 / P), [fR, fT, fP]),
                "B": (T, [fT]),
                "n": (P, [fP]),
            },
            [fR, fT, fP],
        )

    return wrapper


def _de_levie_finite(mdl: _MdlElement) -> Element:
    # Z = (R / L^(1/2)) * coth(L^(1/2)) where L = 1/A + B*(I*w)^Phi
    # Fixing R_r at 1 makes the argument of coth independent of whether it is
    # implemented as d*sqrt(R_i*R_r) or as d*sqrt(R_i/R_r).
    A, fA = mdl.get(0)
    B, fB = mdl.get(1)
    Phi, fPhi = mdl.get(2)
    R, fR = mdl.get(3)
    return _create(
        DeLevieFiniteLength,
        mdl,
        {
            "R_i": (R**2 * A, [fR, fA]),
            "R_r": (float64(1.0), []),
            "Y": (A * B, [fA, fB]),
            "n": (Phi, [fPhi]),
            "d": (1 / (R * A), [fR, fA]),
        },
        [fA, fB, fPhi, fR],
    )


def _de_levie_semiinfinite(mdl: _MdlElement) -> Element:
    # Z = 1 / sqrt(1/A + B*(I*w)^Phi), which is a Gerischer element when Phi = 1
    A, fA = mdl.get(0)
    B, fB = mdl.get(1)
    Phi, fPhi = mdl.get(2)
    if Phi != 1.0:
        raise ParsingError(
            f"Expected Phi to be 1 instead of {Phi} for {mdl} since there is no equivalent element in pyimpspec for other values!"
        )

    return _create(
        Gerischer,
        mdl,
        {
            "Y": (sqrt(B), [fB]),
            "k": (1 / (A * B), [fA, fB]),
            "n": (float64(0.5), []),
        },
        [fA, fB, fPhi],
    )


_ZVIEW_ELEMENTS: Dict[int, Callable[[_MdlElement], Element]] = {
    1: _resistor,
    2: _capacitor,
    3: _inductor,
    11: _cpe,
    12: _gerischer,
    13: _qpe,
    21: _warburg(WarburgShort),
    22: _warburg(WarburgOpen),
    23: _de_levie_finite,
    24: _de_levie_semiinfinite,
}


def _close(is_parallel: bool, items: List[Union[Element, Connection]]) -> Connection:
    Class: Type[Connection] = Parallel if is_parallel else Series
    if len(items) == 0:
        raise ConnectionWithoutElements(Class)
    elif is_parallel and len(items) < 2:
        raise InsufficientElementsInParallelConnection()

    return Class(items)


def _build(elements: List[_MdlElement]) -> Circuit:
    stack: List[Tuple[bool, List[Union[Element, Connection]]]] = [(False, [])]

    mdl: _MdlElement
    for mdl in elements:
        if mdl.type in (_BEGIN_PARALLEL, _BEGIN_SERIES):
            stack.append((mdl.type == _BEGIN_PARALLEL, []))
        elif mdl.type in (_END_PARALLEL, _END_SERIES):
            if len(stack) < 2 or stack[-1][0] != (mdl.type == _END_PARALLEL):
                raise ParsingError(f"Unexpected end of connection at {mdl}")
            connection: Connection = _close(*stack.pop())
            stack[-1][1].append(connection)
        elif mdl.type in _ZVIEW_ELEMENTS:
            with errstate(divide="ignore", invalid="ignore", over="ignore"):
                stack[-1][1].append(_ZVIEW_ELEMENTS[mdl.type](mdl))
        elif mdl.type in _UNSUPPORTED_ELEMENTS:
            raise ParsingError(
                f"Unsupported ZView element ({_UNSUPPORTED_ELEMENTS[mdl.type]}): {mdl}"
            )
        else:
            raise ParsingError(f"Unknown ZView element type: {mdl}")

    if len(stack) > 1:
        raise ParsingError("Expected all connections to be closed!")

    return Circuit(_close(*stack.pop()))  # type: ignore


def parse_mdl(path: Union[str, Path]) -> Circuit:
    """
    Generate a Circuit instance from an equivalent circuit model created in ZView (.mdl file).

    The following ZView elements are supported: R, C, L, CPE, QPE, GE, Ws, Wo, Ls, and Lo (only if Phi = 1).
    Parameters are converted when their definitions differ (e.g., the T parameter of a QPE becomes Y = T^n), and fixed parameters remain fixed.
    The names of the elements are used as labels.

    Parameters
    ----------
    path: Union[str, Path]
        The path to the .mdl file.

    Returns
    -------
    |Circuit|
    """
    if not isinstance(path, (str, Path)):
        raise TypeError(f"Expected a string or Path instead of {path=}")
    elif not isinstance(path, Path):
        path = Path(path)

    return _build(_tokenize(_read_lines(path)))
