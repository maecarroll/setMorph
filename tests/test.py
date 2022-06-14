#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from setmorph import exponence
import pandas as pd
from pathlib import Path

here = Path(__file__)


def feature_print(feats):
    return "{" + "}, {".join([", ".join(f) for f in feats]) + "}"


class testSetMorph(unittest.TestCase):

    def test_exponence_description(self):
        fr_toy_cells = {frozenset({'1sg', 'prs'}), frozenset({'fut', '1sg'}),
                        frozenset({'fut', '2pl'}),
                        frozenset({'pst', '1sg'}), frozenset({'fut', '3pl'}),
                        frozenset({'pst', '1pl'}),
                        frozenset({'3sg', 'pst'}), frozenset({'1pl', 'prs'}),
                        frozenset({'2sg', 'fut'}),
                        frozenset({'3sg', 'prs'}), frozenset({'ptcp', 'prs'}),
                        frozenset({'2pl', 'pst'}),
                        frozenset({'2pl', 'prs'}), frozenset({'fut', '1pl'}),
                        frozenset({'3pl', 'pst'}),
                        frozenset({'2sg', 'pst'}), frozenset({'inf'}),
                        frozenset({'3pl', 'prs'}),
                        frozenset({'fut', '3sg'}), frozenset({'2sg', 'prs'})}

        feature_structure = {
            'Tense': {frozenset({'prs'}), frozenset({'fut'}), frozenset({'pst'})},
            'Mode': {frozenset({'ptcp'}), frozenset({'inf'})},
            'PersonNumber': {frozenset({'1sg'}), frozenset({'2sg'}), frozenset({'3sg'}),
                             frozenset({'1pl'}), frozenset({'2pl'}), frozenset({'3pl'})}
            }

        values = [
            # When the distribution is identical to the cells, there are no exponents
            (fr_toy_cells, fr_toy_cells, feature_structure, set()),

            # When some cells can be reduced to a feature, they are
            (fr_toy_cells, {frozenset({'3pl', 'pst'}), frozenset({'fut', '1sg'}),
                            frozenset({'fut', '2pl'}), frozenset({'inf'}),
                            frozenset({'fut', '3pl'}), frozenset({'fut', '3sg'}),
                            frozenset({'2sg', 'fut'}), frozenset({'fut', '1pl'})},
             feature_structure,
             {frozenset({'inf'}), frozenset({'fut'}), frozenset({'3pl', 'pst'})},),

            # When the distribution can not be reduced, it is not
            (fr_toy_cells,
             {frozenset({'1pl', 'prs'}), frozenset({'fut', '3pl'}),
              frozenset({'fut', '1pl'}), frozenset({'3pl', 'prs'})},
             feature_structure,
             {frozenset({'1pl', 'prs'}), frozenset({'fut', '3pl'}),
              frozenset({'1pl', 'fut'}), frozenset({'prs', '3pl'})}
             ),

            # When a both a value (prs) and its subset (ptcp) can describe the distr,
            # only the super set is kept
            (fr_toy_cells,
             {frozenset({'1sg', 'prs'}),
              frozenset({'1pl', 'prs'}),
              frozenset({'3sg', 'prs'}),
              frozenset({'ptcp', 'prs'}),
              frozenset({'2pl', 'prs'}),
              frozenset({'3pl', 'prs'}),
              frozenset({'2sg', 'prs'})},
             feature_structure,
             {frozenset({'prs'})}
             ),

            # When the distribution fills a dimension,
            # it is not an exponent of this dimension !
            # (this is 3x3 paradigm + inf, with 3x3 in dista)
            ({frozenset({'1sg', 'prs'}), frozenset({'fut', '1sg'}),
              frozenset({'pst', '1sg'}),
              frozenset({'3sg', 'pst'}),
              frozenset({'2sg', 'fut'}),
              frozenset({'3sg', 'prs'}),
              frozenset({'2sg', 'pst'}), frozenset({'inf'}),
              frozenset({'fut', '3sg'}), frozenset({'2sg', 'prs'})},
             {frozenset({'1sg', 'prs'}), frozenset({'fut', '1sg'}),
              frozenset({'pst', '1sg'}),
              frozenset({'3sg', 'pst'}),
              frozenset({'2sg', 'fut'}),
              frozenset({'3sg', 'prs'}),
              frozenset({'2sg', 'pst'}),
              frozenset({'fut', '3sg'}), frozenset({'2sg', 'prs'})},
             {
                 'Tense': {frozenset({'prs'}), frozenset({'fut'}), frozenset({'pst'})},
                 'Mode': {frozenset({'ptcp'}), frozenset({'inf'})},
                 'PersonNumber': {frozenset({'1sg'}), frozenset({'2sg'}),
                                  frozenset({'3sg'})}
             },
             set()
             ),

        ]
        for cells, dista, fs, res in values:
            d = exponence(cells, dista, fs)
            print("res:", d, type(d))
            print("exp:", res, type(res))
            if res is None:
                self.assertEqual(d, res)
            else:
                self.assertSetEqual(d, res)

        ## What if we have a paradigm like:


if __name__ == '__main__':
    unittest.main()
