#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from setmorph import exponence
from pathlib import Path
from hypothesis import given, note, example
from itertools import combinations, chain
from strategies import cells_dist_feats
import pandas as pd

here = Path(__file__)

ex_sg_1 = pd.DataFrame([["a", "s", 0, "x", {("SG", "1")}, {("SG", "1")}],
                        ["a", "s", 1, "y", {("SG", "2")}, {("SG", "2")}],
                        ["a", "s", 2, "z", {("SG", "3")}, {("SG", "3")}],
                        ], columns=["lexeme", "tier", "slot", "formative", "dist_a",
                                    "exponence"])

ex_sg_2 = pd.DataFrame(
    [["a", "s", 0, "x", {("SG", "1"), ("SG", "2")}, {("SG", "1"), ("SG", "2")}],
     ["a", "s", 1, "y", {("SG", "2"), ("SG", "3")}, {("SG", "2"), ("SG", "3")}]],
    columns=["lexeme", "tier", "slot", "formative", "dist_a",
             "exponence"])

ex_sg_3 = pd.DataFrame(
    [["a", "s", 0, "x", {("SG", "1"), ("SG", "2"), ("SG", "3")}, {("SG",)}],
     ["a", "s", 1, "y", {("SG", "3")}, {("SG", "3")}]],
    columns=["lexeme", "tier", "slot", "formative", "dist_a", "exponence"])

ex_sg_4 = pd.DataFrame(
    [["a", "s", 0, "x", {("SG", "1"), ("SG", "2")}, {("SG", "1"), ("SG", "2")}],
     ["a", "s", 2, "y", {("SG", "3")}, {("SG", "3")}],
     ["a", "s", 2, "z", {("SG", "3")}, {("SG", "3")}],
     ], columns=["lexeme", "tier", "slot", "formative", "dist_a", "exponence"])


class testAllomorphy(unittest.TestCase):

    @example(ex_sg_1)
    @example(ex_sg_2)
    @example(ex_sg_3)
    @example(ex_sg_4)
    def test_x(self, exponents):
        allomorphs = classify_allomorphy(exponents)
        # the number of formatives which have v in their delta is more than 1




if __name__ == '__main__':
    unittest.main()
