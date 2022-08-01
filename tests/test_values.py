#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from setmorph import find_exponents, values_per_word, values_table, \
    classify_allomorphy, count_formatives
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

    @given(exponents_df())
    @example((ex_sg_a, sg))
    @example((ex_sg_b, sg))
    @example((ex_sg_c, sg))
    @example((ex_sg_d, sg))
    @example((ex_sg_e, sg))
    @settings(deadline=None)
    def test_count_formatives_inplace(self, args):
        """Tests counting formatives"""
        df, features = args
        exps = find_exponents(df, features)
        values_words = values_per_word(df, exps)
        values = values_table(values_words)
        count_formatives(values)
        note(values)

        # changed it in place
        self.assertIn("# formatives", list(values.columns))

    @given(exponents_df())
    @example((ex_sg_a, sg))
    @example((ex_sg_b, sg))
    @example((ex_sg_c, sg))
    @example((ex_sg_d, sg))
    @example((ex_sg_e, sg))
    @settings(deadline=None)
    def test_count_formatives_positive_integers(self, args):
        """Tests counting formatives"""
        df, features = args
        exps = find_exponents(df, features)
        values_words = values_per_word(df, exps)
        values = values_table(values_words)
        count_formatives(values)
        note(values)

        self.assertTrue((values["# formatives"].apply(type) == int).all())
        self.assertTrue((values["# formatives"] > 0).all())

    #
    # @given(exponents_df())
    # @example((ex_sg_a, sg))
    # @example((ex_sg_b, sg))
    # @example((ex_sg_c, sg))
    # @example((ex_sg_d, sg))
    # @example((ex_sg_e, sg))
    # def test_allomorphy_shape(self, args):
    #     """Tests the expected shape of the allomorphy dataframe"""
    #     df, features = args
    #     exps = find_exponents(df, features)
    #     res = classify_allomorphy(exps, df)
    #
    #     note(", ".join(list(res.columns)))
    #     exp_cols = ['lexeme', 'vals', 'allomorph set', 'allomorph set count',
    #                 'cells with v',
    #                 '% allomorphs to cells containing v']
    #
    #     # Expected columns
    #     self.assertListEqual(list(res.columns), exp_cols)
    #
    #     # sets of formatives
    #     # Formatives are tuples of: tier, slot, sounds
    #     def contains_formatives_triples(elt):
    #         self.assertTrue(type(elt) is set)
    #         for fs in elt:
    #             for f in fs:
    #                 self.assertTrue(len(f) == 3)
    #                 self.assertTrue(type(f[0]) is str)
    #                 self.assertTrue(type(f[1]) is int)
    #                 self.assertTrue(type(f[2]) is str)
    #
    #     res['allomorph set'].apply(contains_formatives_triples)
    #
    # @given(exponents_df())
    # @example((ex_sg_a, sg))
    # @example((ex_sg_b, sg))
    # @example((ex_sg_c, sg))
    # @example((ex_sg_d, sg))
    # @example((ex_sg_e, sg))
    # @settings(deadline=None)
    # def test_allomorphy_same_value(self, args):
    #     """Tests that all formatives have the value in their exponent set."""
    #     df, features = args
    #     exps = find_exponents(df, features)
    #     res = classify_allomorphy(exps, df)
    #     note("Allomorphy:\n\t" + str(res))
    #     vals = exps.set_index(["lexeme", "tier", "slot", "formative"])["exponence"] \
    #         .apply(lambda v: set(chain(*v))).to_dict()
    #
    #     note(exps)
    #     res["fs"] = res["allomorph set"].apply(lambda x: list(chain(*x)))
    #     res = res.explode("fs")
    #
    #     def same_value(row):
    #         t, s, f = row["fs"]
    #         recovered_values = vals[(row.lexeme, t, s, f)]
    #         self.assertIn(row.vals, recovered_values)
    #
    #     res.apply(same_value, axis=1)
    #
    # @settings(deadline=None)
    # @example((ex_sg_a, sg))
    # @example((ex_sg_b, sg))
    # @example((ex_sg_c, sg))
    # @example((ex_sg_d, sg))
    # @example((ex_sg_e, sg))
    # @given(exponents_df())
    # def test_allomorphy_diff_words(self, args):
    #     """Test that there are at least two formatives from different words."""
    #     df, features = args
    #     exps = find_exponents(df, features)
    #     res = classify_allomorphy(exps, df)
    #
    #     words = df.groupby(["lexeme", "tier", "slot", "formative"]) \
    #         .apply(lambda g: [(r.cell, r.form) for i, r in g.iterrows()]) \
    #         .to_dict()
    #
    #     def has_different_words(row):
    #         w = None
    #         allomorphs = set(chain(*r["allomorph set"]))
    #         for t, s, f in allomorphs:
    #             new = words[(row.lexeme, t, s, f)]
    #             if w is None:
    #                 w = new
    #             elif w != new:
    #                 return True
    #         return False
    #
    #     for i, r in res.iterrows():
    #         self.assertTrue(has_different_words(r))
    #
    # @settings(deadline=None)
    # @example((ex_sg_a, sg))
    # @example((ex_sg_b, sg))
    # @example((ex_sg_c, sg))
    # @example((ex_sg_d, sg))
    # @example((ex_sg_e, sg))
    # @given(exponents_df())
    # def test_allomorphy_overall_count(self, args):
    #     """Test that there are the right number of allomorphic values."""
    #     df, features = args
    #     exps = find_exponents(df, features)
    #     res = classify_allomorphy(exps, df)
    #
    #     # This is an alternate allomorphy implementation, which provides less info,
    #     # is slower
    #     # but more straightforward
    #
    #     df = pd.merge(df, exps, on=["lexeme", "tier", "slot", "formative"],
    #                   how="left")
    #
    #     note(str(df))
    #
    #     val_to_word_count = defaultdict(lambda: defaultdict(set))
    #
    #     # Counts allomorphs by constructing a dict of :
    #     # (lexeme, value) => (formative) => set of (cell, form)
    #     for i, row in df.iterrows():
    #         lex = row.lexeme
    #         f = row.form
    #         c = row.cell
    #         a = (row.tier, row.slot, row.formative)
    #         vals_here = {vs for vs in row.exponence if vs <= row.celllist}
    #         for val in chain(*vals_here):
    #             val_to_word_count[(lex, val)][a].add((c, f))
    #
    #     expected_size = 0
    #
    #     # For each lexeme, value pair
    #     for key in val_to_word_count:
    #         # How many distinct word sets did we find ?
    #         word_sets = set(map(lambda x: tuple(sorted(x)),
    #                             val_to_word_count[key].values()))
    #         # We should find one row if there were several word sets
    #         expected_size += int(len(word_sets) > 1)
    #
    #     note(res)
    #     note(str(val_to_word_count))
    #     self.assertEqual(res.shape[0], expected_size)

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


# So far allomorphy would work if returned empty table. Count exp number of values ?


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
