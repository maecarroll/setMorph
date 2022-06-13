#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from setmorph import shortest
import pandas as pd
from pathlib import Path

here = Path(__file__)


def feature_print(feats):
    return "{"+ "}, {".join([", ".join(f) for f in feats]) + "}"

class testSetMorph(unittest.TestCase):

    def test_shortest_description(self):
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

        values = [
            # When the distribution is identical to the cells, it returns None
            (fr_toy_cells, fr_toy_cells, set()),

            # When some cells can be reduced to a feature, they are
            (fr_toy_cells, {frozenset({'3pl', 'pst'}), frozenset({'fut', '1sg'}),
                            frozenset({'fut', '2pl'}), frozenset({'inf'}),
                            frozenset({'fut', '3pl'}), frozenset({'fut', '3sg'}),
                            frozenset({'2sg', 'fut'}), frozenset({'fut', '1pl'})},
             {frozenset({'inf'}), frozenset({'fut'}), frozenset({'3pl', 'pst'})},),

            # When the distribution can not be reduced, it is not
            (fr_toy_cells,
             {frozenset({'1pl', 'prs'}), frozenset({'fut', '3pl'}),
              frozenset({'fut', '1pl'}), frozenset({'3pl', 'prs'})},
             {frozenset({'1pl', 'prs'}), frozenset({'fut', '3pl'}),
              frozenset({'1pl', 'fut'}), frozenset({'prs', '3pl'})}
             ),

             # Several reductions
            (fr_toy_cells,
             {frozenset({'1sg', 'prs'}), frozenset({'fut', '1sg'}),
              frozenset({'fut', '2pl'}),
              frozenset({'pst', '1sg'}), frozenset({'fut', '3pl'}),
              frozenset({'pst', '1pl'}),
              frozenset({'3sg', 'pst'}), frozenset({'1pl', 'prs'}),
              frozenset({'2sg', 'fut'}),
              frozenset({'3sg', 'prs'}), frozenset({'ptcp', 'prs'}),
              frozenset({'2pl', 'pst'}),
              frozenset({'2pl', 'prs'}), frozenset({'fut', '1pl'}),
              frozenset({'3pl', 'pst'}),
              frozenset({'2sg', 'pst'}),
              frozenset({'3pl', 'prs'}),
              frozenset({'fut', '3sg'}), frozenset({'2sg', 'prs'})},
             {frozenset({'fut'}), frozenset({'prs'}), frozenset({'pst'}),
              frozenset({'1sg'}), frozenset({'2sg'}), frozenset({'3sg'}),
              frozenset({'1pl'}), frozenset({'2pl'}), frozenset({'3pl'}),}
             ),

            # When several reductions are possible and equivalent,
            # We return all of them
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
             {frozenset({'fut'}), frozenset({'prs'}), frozenset({'pst'}),
              frozenset({'1sg'}), frozenset({'2sg'}), frozenset({'3sg'})}
        ),


        ]
        for cells, dista, res in values:
            d = shortest(cells, dista)
            if res is None:
                self.assertEqual(d, res)
            else:
                self.assertSetEqual(d, res)

        ## What if we have a paradigm like:


if __name__ == '__main__':
    unittest.main()
