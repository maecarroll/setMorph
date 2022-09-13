#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from setmorph import get_exponents, get_reals, get_real_per_word, classify_allomorphy
from pathlib import Path
from hypothesis import given, note, example, settings
from .strategies import exponents_df
from itertools import chain
from collections import defaultdict
import pandas as pd

here = Path(__file__)

sg = {"Person": {frozenset({"1"}), frozenset({"2"}), frozenset({"3"})},
      "Number": {frozenset({"SG"}), frozenset({"PL"})}
      }

ex_sg_a = pd.DataFrame(
    [["l1", "x", "segmental", 0, "x", frozenset({"SG", "1"})],
     ["l1", "y", "segmental", 0, "y", frozenset({"SG", "2"})],
     ["l1", "z", "segmental", 0, "z", frozenset({"SG", "3"})],
     ["l1", "w", "segmental", 0, "w", frozenset({"PL", "1"})],
     ["l1", "w", "segmental", 0, "w", frozenset({"PL", "2"})],
     ["l1", "w", "segmental", 0, "w", frozenset({"PL", "3"})],

     ],
    columns=["lexeme", "form", "tier", "slot", "formative", "cell"])

ex_sg_b = pd.DataFrame(
    [["l1", "x", "segmental", 0, "x", frozenset({"SG", "1"})],
     ["l1", "x y", "segmental", 0, "x", frozenset({"SG", "2"})],
     ["l1", "x y", "segmental", 1, "y", frozenset({"SG", "2"})],
     ["l1", "y", "segmental", 0, "y", frozenset({"SG", "3"})],
     ["l1", "z", "segmental", 0, "z", frozenset({"PL", "1"})],
     ["l1", "z", "segmental", 0, "z", frozenset({"PL", "2"})],
     ["l1", "z", "segmental", 0, "z", frozenset({"PL", "3"})],

     ],
    columns=["lexeme", "form", "tier", "slot", "formative", "cell"])

ex_sg_c = pd.DataFrame(
    [["l1", "x", "segmental", 0, "x", frozenset({"SG", "1"})],
     ["l1", "x", "segmental", 0, "x", frozenset({"SG", "2"})],
     ["l1", "x y", "segmental", 0, "x", frozenset({"SG", "3"})],
     ["l1", "x y", "segmental", 1, "y", frozenset({"SG", "3"})],
     ["l1", "z", "segmental", 0, "z", frozenset({"PL", "1"})],
     ["l1", "z", "segmental", 0, "z", frozenset({"PL", "2"})],
     ["l1", "z", "segmental", 0, "z", frozenset({"PL", "3"})],

     ],
    columns=["lexeme", "form", "tier", "slot", "formative", "cell"])

ex_sg_d = pd.DataFrame(
    [["l1", "x", "segmental", 0, "x", frozenset({"SG", "1"})],
     ["l1", "x", "segmental", 0, "x", frozenset({"SG", "2"})],
     ["l1", "y z", "segmental", 0, "y", frozenset({"SG", "3"})],
     ["l1", "y z", "segmental", 1, "z", frozenset({"SG", "3"})],
     ["l1", "w", "segmental", 0, "w", frozenset({"PL", "1"})],
     ["l1", "w", "segmental", 0, "w", frozenset({"PL", "2"})],
     ["l1", "w", "segmental", 0, "w", frozenset({"PL", "3"})],

     ],
    columns=["lexeme", "form", "tier", "slot", "formative", "cell"])

ex_sg_e = pd.DataFrame(
    [
        ["l1", "x y", "segmental", 0, "x", frozenset({"SG", "1"})],
        ["l1", "x y", "segmental", 1, "y", frozenset({"SG", "1"})],
        ["l1", "x y", "segmental", 0, "x", frozenset({"SG", "2"})],
        ["l1", "x y", "segmental", 1, "y", frozenset({"SG", "2"})],
        ["l1", "x", "segmental", 0, "x", frozenset({"SG", "3"})],
        ["l1", "z", "segmental", 0, "z", frozenset({"PL", "1"})],
        ["l1", "z", "segmental", 0, "z", frozenset({"PL", "2"})],
        ["l1", "y", "segmental", 0, "y", frozenset({"PL", "3"})],

    ],
    columns=["lexeme", "form", "tier", "slot", "formative", "cell"])

ex_ve_1 = pd.DataFrame([
    ["beb", "b e b a b", "segmental", 0, "be", frozenset({"b"})],
    ["beb", "b e b a b", "segmental", 1, "bab", frozenset({"b"})],
    ["beb", "b a b", "segmental", 0, "bab", frozenset({"b"})],
    ["beb", "b e", "segmental", 0, "be", frozenset({"e"})]
],
    columns=["lexeme", "form", "tier", "slot", "formative", "cell"]
)
ex_ve_2 = pd.DataFrame([
    ["bab", "b a", "segmental", 0, "ba", frozenset({"b"})],
    ["bab", "b a b a b b e c", "segmental", 0, "ba", frozenset({"e"})],
    ["bab", "b a b a b b e c", "segmental", 1, "bab", frozenset({"e"})],
    ["bab", "b a b a b b e c", "segmental", 2, "bec", frozenset({"e"})]
],
    columns=["lexeme", "form", "tier", "slot", "formative", "cell"]
)
ex_ve_feats = {'A': {frozenset({'b'}), frozenset({'e'})}}


class TestValueClassifications(unittest.TestCase):

    @given(exponents_df())
    @settings(deadline=None)
    def test_reals_shape(self, args):
        """Tests the expected shape of the allomorphy dataframe"""
        df, features = args
        exps = get_exponents(df, features)
        note(exps)
        reals = get_reals(exps)
        note(reals)
        # Expected columns
        self.assertListEqual(list(reals.columns), ['real', '|real|'])

    @given(exponents_df())
    @settings(deadline=None)
    def test_allomorphy_shape(self, args):
        """Tests the expected shape of the allomorphy dataframe"""
        df, features = args
        exps = get_exponents(df, features)
        reals = get_reals(exps)
        real_w = get_real_per_word(df, reals)
        reals = classify_allomorphy(reals, real_w)

        note(", ".join(list(reals.columns)))
        exp_cols = ['real', '|real|', 'allomorphic_sets']

        # Expected columns
        self.assertListEqual(list(reals.columns), exp_cols)

    @given(exponents_df())
    @example((ex_ve_1, ex_ve_feats))
    @example((ex_ve_2, ex_ve_feats))
    @settings(deadline=None)
    def test_real_words_unique_words(self, args):
        """Test that there is a single row for any word"""
        df, features = args
        exponents = get_exponents(df, features)
        reals = get_reals(exponents)
        real_w = get_real_per_word(df, reals)
        self.assertFalse(
            real_w[["lexeme", "cell", "vals", "form"]].duplicated().any())


if __name__ == '__main__':
    unittest.main()
