#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from setmorph import find_exponents, classify_cumulation, count_elts
from pathlib import Path
from hypothesis import given, note, settings
from .strategies import exponents_df

here = Path(__file__)


class testFormativeClassifications(unittest.TestCase):

    @given(exponents_df())
    @settings(deadline=None)
    def test_cumulation_changes_inplace(self, args):
        """Tests that classification into cumulative exponents adds proper columns."""
        df, features = args
        exps = find_exponents(df, features)
        max_dimensions = df["celllist"].fillna("").apply(len).max()

        classify_cumulation(exps, max_dimensions)
        note(f"Exponents:\n{exps}")

        # Expected columns are present
        expected_cols = ['lexeme', 'tier', 'slot', 'formative',
                         'exponence', '# of cells', 'dist_a',
                         'cumulative cells', 'longest cumulation',
                         '% cells cumulative', '% dimensions cumulation']
        self.assertListEqual(list(exps.columns), expected_cols)

    @given(exponents_df())
    @settings(deadline=None)
    def test_cumulation_percentages(self, args):
        """Tests that percentages for cumulation are between 0 and 100"""
        df, features = args
        exps = find_exponents(df, features)
        max_dimensions = df["celllist"].fillna("").apply(len).max()

        classify_cumulation(exps, max_dimensions)
        note(f"Exponents:\n{exps}")

        percent_cmlt = exps['% cells cumulative']
        percent_dims = exps['% dimensions cumulation']
        self.assertTrue(((percent_cmlt <= 100) & (percent_cmlt >= 0)).all())
        self.assertTrue(((percent_dims <= 100) & (percent_dims >= 0)).all())
        # Add test: if >0 cumulative cells, then % must also be > 0

    @given(exponents_df())
    @settings(deadline=None)
    def test_cumulation_total(self, args):
        """Tests that cumulative are subsets of exponent values"""
        df, features = args
        exps = find_exponents(df, features)
        max_dimensions = df["celllist"].fillna("").apply(len).max()

        classify_cumulation(exps, max_dimensions)
        note(f"Exponents:\n{exps}")

        self.assertTrue((exps["cumulative cells"] <= exps["exponence"]).all())

    @given(exponents_df())
    @settings(deadline=None)
    def test_cumulation_longest(self, args):
        """Tests the exact value of the longest cumulation"""
        df, features = args
        exps = find_exponents(df, features)
        max_dimensions = df["celllist"].fillna("").apply(len).max()

        classify_cumulation(exps, max_dimensions)
        note(f"Exponents:\n{exps}")

        # above 1 unless there is none
        self.assertTrue(((exps['longest cumulation'] > 1)
                         | (exps["cumulative cells"] == set()))
                        .all())

        # below or equal to the number of features
        self.assertTrue((exps['longest cumulation'] <= len(features)).all())

    @given(exponents_df())
    @settings(deadline=None)
    def test_cumulation_intent(self, args):
        """Tests that cumulative values are all exponential values of len > 1"""
        df, features = args
        exps = find_exponents(df, features)
        max_dimensions = df["celllist"].fillna("").apply(len).max()

        classify_cumulation(exps, max_dimensions)
        note(f"Exponents:\n{exps}")

        # Cumulative cells are a subset of exponents
        self.assertTrue((exps["exponence"] >= exps["cumulative cells"]).all())

        # Cumulative cells have length > 1
        self.assertTrue((exps["cumulative cells"]
                         .apply(lambda c: len(min(c)) > 1 if c else True)).all())

        # all not in cumulative cells have length 1
        self.assertTrue(((exps["exponence"] - exps["cumulative cells"])
                         .apply(lambda x: len(max(x)) == 1 if x else True))
                        .all())

    @given(exponents_df())
    @settings(deadline=None)
    def test_syn(self, args):
        """Tests that there are between 0 and len(distr) sets"""
        df, features = args
        exps = find_exponents(df, features)
        count_elts(exps, "exponence", "# values")

        self.assertTrue("# values" in exps.columns)
        self.assertTrue((exps["# values"] <= exps["dist_a"].apply(len)).all())
        self.assertTrue((0 <= exps["# values"]).all())


if __name__ == '__main__':
    unittest.main()
