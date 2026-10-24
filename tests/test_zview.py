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


from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Dict,
    List,
    Tuple,
)
from unittest import TestCase
from warnings import catch_warnings
from numpy import (
    allclose,
    logspace,
    pi,
    sqrt,
    tanh,
)
from pyimpspec import (
    Circuit,
    Element,
    parse_mdl,
)
from pyimpspec.exceptions import ParsingError


TEST_DIR: Path = Path(__file__).parent
FREQUENCIES = logspace(6, -3, 91)
s = 1j * 2 * pi * FREQUENCIES

Parameter = Tuple[int, str, str, int]
MdlElement = Tuple[int, str, List[Parameter]]


def _model(elements: List[MdlElement]) -> str:
    lines: List[str] = [
        "ZView Circuit Model:          2.0",
        "End ZView:                    2.0",
        f"Begin Circuit Model:          {len(elements)}",
    ]
    for i, (type_, name, parameters) in enumerate(elements):
        lines.append(f"  Element #{i} Type:            {type_}")
        lines.append(f"  Element #{i} Name:            {name}")
        for j, key, value, free in parameters:
            lines.append(f"  Element #{i},{j} Name:          {key}")
            lines.append(f"  Element #{i},{j} Value:         {value}")
            lines.append(f"  Element #{i},{j} Free:          {free}")
    lines.append(f"End Circuit Model:            {len(elements)}")
    return "\n".join(lines)


def _parse(content: str) -> Circuit:
    with TemporaryDirectory() as directory:
        path: Path = Path(directory) / "model.mdl"
        path.write_text(content)
        return parse_mdl(path)


def _elements(circuit: Circuit) -> Dict[str, Element]:
    return {element.get_label(): element for element in circuit.get_elements()}


class TestZView(TestCase):
    def test_nested(self):
        circuit: Circuit = parse_mdl(str(TEST_DIR / "zview-nested.mdl"))
        self.assertEqual(circuit.to_string(), "[R([RC]C[R(RR)])(RC)]")
        self.assertEqual(
            [element.get_label() for element in circuit.get_elements()],
            ["R1", "R2", "C3", "C1", "R4", "R5", "R6", "R3", "C2"],
        )
        self.assertTrue(all(all(element.are_fixed().values()) for element in circuit.get_elements()))

    def test_supported(self):
        with catch_warnings(record=True) as warnings:
            circuit: Circuit = parse_mdl(TEST_DIR / "zview-supported.mdl")
        self.assertEqual(len(warnings), 0, [str(w.message) for w in warnings])
        self.assertEqual(circuit.to_string(), "[(RR)CLQQGWsWoLsG]")

        elements: Dict[str, Element] = _elements(circuit)
        self.assertEqual(elements["R1"].get_value("R"), 250.0)
        self.assertEqual(elements["C1"].get_value("C"), 1.5e-6)
        self.assertTrue(elements["R2"].is_fixed("R"))
        self.assertFalse(elements["R1"].is_fixed("R"))
        self.assertTrue(elements["L1"].is_fixed("L"))
        self.assertFalse(elements["CPE1"].is_fixed("Y"))
        self.assertTrue(elements["GE1"].is_fixed("n"))
        self.assertFalse(elements["Wo1"].is_fixed("Y"))
        self.assertFalse(elements["Wo1"].is_fixed("B"))
        self.assertTrue(elements["Wo1"].is_fixed("n"))
        self.assertTrue(elements["Ls1"].is_fixed("R_r"))
        self.assertFalse(elements["Ls1"].is_fixed("d"))

        # Impedances calculated using the equations from the ZView documentation.
        Ls_lambda = 1 / 3 + 1e-3 * s**0.9
        expected = {
            "R2": 100.0 + 0 * s,
            "R1": 250.0 + 0 * s,
            "C1": 1 / (s * 1.5e-6),
            "L1": s * 2e-7,
            "CPE1": 1 / (3e-5 * s**0.85),
            "QPE1": 1 / (s * 4e-4) ** 0.9,
            "GE1": 1 / (2e-3 * sqrt(50 + s)),
            "Ws1": 1000 * tanh((s * 0.1) ** 0.45) / (s * 0.1) ** 0.45,
            "Wo1": 500 / tanh((s * 0.2) ** 0.5) / (s * 0.2) ** 0.5,
            "Ls1": 2 / sqrt(Ls_lambda) / tanh(sqrt(Ls_lambda)),
            "Lo1": 1 / sqrt(1 / 3 + 1e-3 * s),
        }
        self.assertEqual(set(elements.keys()), set(expected.keys()))
        for label, Z in expected.items():
            self.assertTrue(
                allclose(elements[label].get_impedances(FREQUENCIES), Z, rtol=1e-10, atol=0),
                label,
            )

        Z_parallel = 1 / (1 / expected.pop("R1") + 1 / expected.pop("R2"))
        self.assertTrue(
            allclose(
                circuit.get_impedances(FREQUENCIES),
                Z_parallel + sum(expected.values()),
                rtol=1e-10,
                atol=0,
            )
        )

    def test_all_elements(self):
        # This file uses the cp1252 encoding and contains elements that are not supported.
        with catch_warnings(record=True):
            with self.assertRaisesRegex(ParsingError, "DE1"):
                parse_mdl(TEST_DIR / "zview-all-elements.mdl")

    def test_unsupported(self):
        type_: int
        for type_ in (92, 93, 94, 95, 96, 97, 100, 200, 999):
            with self.assertRaisesRegex(ParsingError, f"type {type_}"):
                _parse(_model([(type_, "X1", [(0, "R", "1", 0)])]))

        with self.assertRaisesRegex(ParsingError, "Phi"):
            _parse(_model([(24, "Lo1", [(0, "A", "1", 0), (1, "B", "1", 0), (2, "Phi", "0,9", 0)])]))

    def test_invalid_structure(self):
        R: MdlElement = (1, "R1", [(0, "R", "1", 0)])
        with self.assertRaises(ParsingError):
            _parse(_model([(-1, "Begin_Parallel", []), R, R]))
        with self.assertRaises(ParsingError):
            _parse(_model([(-1, "Begin_Parallel", []), R, R, (-4, "End_Series", [])]))
        with self.assertRaises(ParsingError):
            _parse(_model([(-1, "Begin_Parallel", []), R, (-2, "End_Parallel", [])]))
        with self.assertRaises(ParsingError):
            _parse(_model([]))
        with self.assertRaises(ParsingError):
            _parse(_model([R]).replace("Begin Circuit Model:          1", "Begin Circuit Model:          2"))
        with self.assertRaises(ParsingError):
            _parse(_model([R]).replace("Element #0,0 Free", "Element #0,0 Freedom"))
        with self.assertRaises(ParsingError):
            _parse(_model([(1, "R1", [(0, "R", "1", 0)])]).replace("Value:         1", "Value:         one"))
        with self.assertRaises(ParsingError):
            _parse(_model([(1, "R1", [(1, "R", "1", 0)])]))
        with self.assertRaises(TypeError):
            parse_mdl(1)

    def test_warnings(self):
        with self.assertWarnsRegex(UserWarning, "'Free' code"):
            _parse(_model([(1, "R1", [(0, "R", "1", 2)])]))

        # Only R is fixed, but Y depends on all three parameters of Ws.
        with self.assertWarnsRegex(UserWarning, "free parameters"):
            _parse(_model([(21, "Ws1", [(0, "R", "1", 0), (1, "T", "1", 1), (2, "P", "0,5", 0)])]))

        with self.assertWarnsRegex(UserWarning, "default value"):
            _parse(_model([(21, "Ws1", [(0, "R", "0", 0), (1, "T", "1", 0), (2, "P", "0,5", 0)])]))
