#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from setmorph import find_exponents, values_per_word, values_table, \
    classify_allomorphy, count_formatives, Formative
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
    [["l1", "SG.1", "x", "segmental", 0, "x", frozenset({"SG", "1"})],
     ["l1", "SG.2", "y", "segmental", 0, "y", frozenset({"SG", "2"})],
     ["l1", "SG.3", "z", "segmental", 0, "z", frozenset({"SG", "3"})],
     ["l1", "PL.1", "w", "segmental", 0, "w", frozenset({"PL", "1"})],
     ["l1", "PL.2", "w", "segmental", 0, "w", frozenset({"PL", "2"})],
     ["l1", "PL.3", "w", "segmental", 0, "w", frozenset({"PL", "3"})],

     ],
    columns=["lexeme", "cell", "form", "tier", "slot", "formative", "celllist"])

ex_sg_b = pd.DataFrame(
    [["l1", "SG.1", "x", "segmental", 0, "x", frozenset({"SG", "1"})],
     ["l1", "SG.2", "x y", "segmental", 0, "x", frozenset({"SG", "2"})],
     ["l1", "SG.2", "x y", "segmental", 1, "y", frozenset({"SG", "2"})],
     ["l1", "SG.3", "y", "segmental", 0, "y", frozenset({"SG", "3"})],
     ["l1", "PL.1", "z", "segmental", 0, "z", frozenset({"PL", "1"})],
     ["l1", "PL.2", "z", "segmental", 0, "z", frozenset({"PL", "2"})],
     ["l1", "PL.3", "z", "segmental", 0, "z", frozenset({"PL", "3"})],

     ],
    columns=["lexeme", "cell", "form", "tier", "slot", "formative", "celllist"])

ex_sg_c = pd.DataFrame(
    [["l1", "SG.1", "x", "segmental", 0, "x", frozenset({"SG", "1"})],
     ["l1", "SG.2", "x", "segmental", 0, "x", frozenset({"SG", "2"})],
     ["l1", "SG.3", "x y", "segmental", 0, "x", frozenset({"SG", "3"})],
     ["l1", "SG.3", "x y", "segmental", 1, "y", frozenset({"SG", "3"})],
     ["l1", "PL.1", "z", "segmental", 0, "z", frozenset({"PL", "1"})],
     ["l1", "PL.2", "z", "segmental", 0, "z", frozenset({"PL", "2"})],
     ["l1", "PL.3", "z", "segmental", 0, "z", frozenset({"PL", "3"})],

     ],
    columns=["lexeme", "cell", "form", "tier", "slot", "formative", "celllist"])

ex_sg_d = pd.DataFrame(
    [["l1", "SG.1", "x", "segmental", 0, "x", frozenset({"SG", "1"})],
     ["l1", "SG.2", "x", "segmental", 0, "x", frozenset({"SG", "2"})],
     ["l1", "SG.3", "y z", "segmental", 0, "y", frozenset({"SG", "3"})],
     ["l1", "SG.3", "y z", "segmental", 1, "z", frozenset({"SG", "3"})],
     ["l1", "PL.1", "w", "segmental", 0, "w", frozenset({"PL", "1"})],
     ["l1", "PL.2", "w", "segmental", 0, "w", frozenset({"PL", "2"})],
     ["l1", "PL.3", "w", "segmental", 0, "w", frozenset({"PL", "3"})],

     ],
    columns=["lexeme", "cell", "form", "tier", "slot", "formative", "celllist"])

ex_sg_e = pd.DataFrame(
    [
        ["l1", "SG.1", "x y", "segmental", 0, "x", frozenset({"SG", "1"})],
        ["l1", "SG.1", "x y", "segmental", 1, "y", frozenset({"SG", "1"})],
        ["l1", "SG.2", "x y", "segmental", 0, "x", frozenset({"SG", "2"})],
        ["l1", "SG.2", "x y", "segmental", 1, "y", frozenset({"SG", "2"})],
        ["l1", "SG.3", "x", "segmental", 0, "x", frozenset({"SG", "3"})],
        ["l1", "PL.1", "z", "segmental", 0, "z", frozenset({"PL", "1"})],
        ["l1", "PL.2", "z", "segmental", 0, "z", frozenset({"PL", "2"})],
        ["l1", "PL.3", "y", "segmental", 0, "y", frozenset({"PL", "3"})],

    ],
    columns=["lexeme", "cell", "form", "tier", "slot", "formative", "celllist"])

ex_ve_1 = pd.DataFrame([
    ["beb", "b", "b e b a b", "segmental", 0, "be", frozenset({"b"})],
    ["beb", "b", "b e b a b", "segmental", 1, "bab", frozenset({"b"})],
    ["beb", "b", "b a b", "segmental", 0, "bab", frozenset({"b"})],
    ["beb", "e", "b e", "segmental", 0, "be", frozenset({"e"})]
],
    columns=["lexeme", "cell", "form", "tier", "slot", "formative", "celllist"]
)
ex_ve_2 = pd.DataFrame([
    ["bab", "b", "b a", "segmental", 0, "ba", frozenset({"b"})],
    ["bab", "e", "b a b a b b e c", "segmental", 0, "ba", frozenset({"e"})],
    ["bab", "e", "b a b a b b e c", "segmental", 1, "bab", frozenset({"e"})],
    ["bab", "e", "b a b a b b e c", "segmental", 2, "bec", frozenset({"e"})]
],
    columns=["lexeme", "cell", "form", "tier", "slot", "formative", "celllist"]
)
ex_ve_feats = {'A': {frozenset({'b'}), frozenset({'e'})}}


class TestValueClassifications(unittest.TestCase):
    #
    # @given(exponents_df())
    # @example((ex_sg_a, sg))
    # @example((ex_sg_b, sg))
    # @example((ex_sg_c, sg))
    # @example((ex_sg_d, sg))
    # @example((ex_sg_e, sg))
    # @settings(deadline=None)
    # def test_count_formatives_inplace(self, args):
    #     """Tests counting formatives"""
    #     df, features = args
    #     exps = find_exponents(df, features)
    #     values_words = values_per_word(df, exps)
    #     values = values_table(values_words)
    #     count_formatives(values)
    #     note(values)
    #
    #     # changed it in place
    #     self.assertIn("# formatives", list(values.columns))
    #
    # @given(exponents_df())
    # @example((ex_sg_a, sg))
    # @example((ex_sg_b, sg))
    # @example((ex_sg_c, sg))
    # @example((ex_sg_d, sg))
    # @example((ex_sg_e, sg))
    # @settings(deadline=None)
    # def test_count_formatives_positive_integers(self, args):
    #     """Tests counting formatives"""
    #     df, features = args
    #     exps = find_exponents(df, features)
    #     values_words = values_per_word(df, exps)
    #     values = values_table(values_words)
    #     count_formatives(values)
    #     note(values)
    #
    #     self.assertTrue((values["# formatives"].apply(type) == int).all())
    #     self.assertTrue((values["# formatives"] > 0).all())
    #

    @given(exponents_df())
    @example((ex_sg_a, sg))
    @example((ex_sg_b, sg))
    @example((ex_sg_c, sg))
    @example((ex_sg_d, sg))
    @example((ex_sg_e, sg))
    @settings(deadline=None)
    def test_allomorphy_shape(self, args):
        """Tests the expected shape of the allomorphy dataframe"""
        df, features = args
        exps = find_exponents(df, features)
        values_words = values_per_word(df, exps)
        values = values_table(values_words)
        classify_allomorphy(df, values)

        note(", ".join(list(values.columns)))
        exp_cols = ['lexeme', 'value', 'formative', 'formatives_by_word',
                     '# allomorph sets', '# words',
                    '% allomorph to words']

        # Expected columns
        self.assertListEqual(list(values.columns), exp_cols)

        # sets of formatives
        # Formatives are tuples of: tier, slot, sounds
        def contains_formatives_triples(elt):
            self.assertTrue(type(elt) is set)
            for f in elt:
                self.assertTrue(type(f) is Formative)

        values['formative'].apply(contains_formatives_triples)

    @given(exponents_df())
    @example((ex_sg_a, sg))
    @example((ex_sg_b, sg))
    @example((ex_sg_c, sg))
    @example((ex_sg_d, sg))
    @example((ex_sg_e, sg))
    @settings(deadline=None)
    def test_values_same_value(self, args):
        """Tests that all formatives have the value in their exponent set."""
        df, features = args
        exps = find_exponents(df, features)
        values_words = values_per_word(df, exps)
        res = values_table(values_words)
        note("Allomorphy:\n\t" + str(res))
        vals = exps.set_index(["lexeme", "tier", "slot", "formative"])["exponence"] \
            .apply(lambda v: set(chain(*v))).to_dict()

        note(exps)
        note(res)
        res["fs"] = res["formatives_by_word"].apply(lambda x: list(chain(*x)))
        res = res.explode("fs")

        def same_value(row):
            t, s, f = row["fs"]
            recovered_values = vals[(row.lexeme, t, s, f)]
            self.assertIn(row.value, recovered_values)

        res.apply(same_value, axis=1)

    @given(exponents_df())
    @example((ex_ve_1, ex_ve_feats))
    @example((ex_ve_2, ex_ve_feats))
    @settings(deadline=None)
    def test_values_words_occur_in_same_word(self, args):
        """Test that ..."""
        df, features = args
        exps = find_exponents(df, features)
        res = values_per_word(df, exps)
        note(exps)
        note(res)

        def exp_list(group):
            return set(group[["tier", "slot", "formative"]].apply(tuple, axis=1))

        # Build a dict of lexeme, cell => list [ {(formative tuple) }, {}]
        words = df.groupby(["lexeme", "cell", "form"]) \
            .apply(exp_list) \
            .groupby(["lexeme", "cell"]) \
            .apply(list) \
            .to_dict()

        def check_word(row):
            expected = words[(row.lexeme, row.cell)]
            note(expected)
            self.assertTrue(any(set(row.formative) <= e for e in expected))

        res.apply(check_word, axis=1)

    @given(exponents_df())
    @example((ex_ve_1, ex_ve_feats))
    @example((ex_ve_2, ex_ve_feats))
    @settings(deadline=None)
    def test_values_words_unique_words(self, args):
        """Test that there is a single row for any word"""
        df, features = args
        exps = find_exponents(df, features)
        res = values_per_word(df, exps)
        note(res)
        self.assertFalse(res[["lexeme", "cell", "value", "form"]].duplicated().any())



if __name__ == '__main__':
    #
    # for i, df in zip("abcde",
    #               [ex_sg_a, ex_sg_b, ex_sg_c, ex_sg_d, ex_sg_e]):
    #     print(f"Example {i}")
    #     print(df)
    #     exps = find_exponents(df, sg)
    #     print(exps)
    #     res = classify_allomorphy(exps, df)
    #     print("Allomorphy:")
    #     print(res)
    #     print(res[res.vals=="SG"]["allomorph set"].iloc[0])

    unittest.main()
