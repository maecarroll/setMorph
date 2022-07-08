#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from setmorph import find_exponents, classify_unique
from pathlib import Path
from hypothesis import given, note, example
from strategies import exponents_df

here = Path(__file__)


class testValueClassifications(unittest.TestCase):

    @given(exponents_df())
    def test_unique(self, args):
        df, features = args
        exps = find_exponents(df, features)
        res = classify_unique(exps)

        # Minimum number of rows is 0
        # Maximum number of rows is the number of values * number of lexemes
        lex = df.lexeme.unique()
        vals = set(v for f in features for v in features[f])

        l = res.shape[0]
        self.assertTrue(l <= len(lex) * len(vals))

        # No duplicate rows
        self.assertFalse(res.duplicated().any())

        # No duplicate lexeme, value couple
        self.assertFalse(res[["lexeme", "value"]].duplicated().any())

if __name__ == '__main__':
    unittest.main()
