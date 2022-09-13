#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from setmorph import get_exponents, classify_cumulation
from pathlib import Path
from hypothesis import given, note, settings
from .strategies import exponents_df

here = Path(__file__)

class testCumulation(unittest.TestCase):

    @given(exponents_df())
    @settings(deadline=None)
    def test_cumulation_changes_inplace(self, args):
        """Tests that classification into cumulative exponents adds proper columns."""
        df, features = args
        exps = get_exponents(df, features)
        classify_cumulation(exps)
        note(f"Exponents:\n{exps}")
        # Expected columns are present
        expected_cols = ['lexeme', 'tier', 'slot', 'formative',
                         'dist', 'exponence', 'vals',
                        '|vals|', '|exp|',
                         'cumulative', 'cumulative_cells',
                        'max_cumulation', 'maximum_possible_cumulation']
        self.assertListEqual(list(exps.columns), expected_cols)

    @given(exponents_df())
    @settings(deadline=None)
    def test_cumulation_total(self, args):
        """Tests that cumulative are subsets of exponent values"""
        df, features = args
        exps = get_exponents(df, features)
        classify_cumulation(exps)
        note(f"Exponents:\n{exps}")

        self.assertTrue((exps["cumulative"] <= exps["exponence"]).all())
        self.assertTrue((exps["cumulative_cells"] <= exps["dist"]).all())

    @given(exponents_df())
    @settings(deadline=None)
    def test_cumulation_longest(self, args):
        """Tests the exact value of the longest cumulation"""
        df, features = args
        exps = get_exponents(df, features)
        classify_cumulation(exps)
        note(f"Exponents:\n{exps}")

        # above 1 unless there is none
        self.assertTrue(((exps['max_cumulation'] > 1)
                         | (exps["cumulative"] == set()))
                        .all())

        # There are cumulative cels if there are cumulative f-v
        self.assertTrue(((exps['max_cumulation'] == 0)
                         | (exps["cumulative_cells"].apply(len) > 0))
                        .all())

        # below or equal to the max
        self.assertTrue((exps['max_cumulation'] <= exps['maximum_possible_cumulation']).all())

    @given(exponents_df())
    @settings(deadline=None)
    def test_cumulation_complement(self, args):
        """Tests that all not in cumulative have len 1"""
        df, features = args
        exps = get_exponents(df, features)
        classify_cumulation(exps)
        note(f"Exponents:\n{exps.apply(str, axis=1)}")

        # all not in cumulative cells have length 1
        self.assertTrue(((exps["exponence"] - exps["cumulative"])
                         .apply(lambda x: len(max(x)) == 1 if x else True))
                        .all())


if __name__ == '__main__':
    unittest.main()
