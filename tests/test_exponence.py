#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from setmorph import exponence
from pathlib import Path
from hypothesis import given, note
from itertools import combinations, chain
from strategies import cells_dist_feats

here = Path(__file__)

class testExponence(unittest.TestCase):

    @given(cells_dist_feats().filter(lambda args: args[0] != args[1]))
    def test_features_are_reduced(self, args):
        """ Tests that if some features can be reduced, they are."""
        (cells, dista, feature_structures) = args
        delta_a = exponence(cells, dista, feature_structures)
        note(f"exp = {delta_a}")

        all_combos = chain(*[combinations(c, i) for c in cells for i in range(len(c)-1)])

        for s in all_combos:
            s = frozenset(s)
            span_word = set(filter(lambda c: s <= c, cells))
            span_dista = {c for c in dista if s <= c}
            # If a set of cells can be reduced to some combo
            if span_word == span_dista:
                # then the set of cells are not in delta
                self.assertFalse(span_dista <= delta_a)

    @given(cells_dist_feats().filter(lambda args: args[0] != args[1]))
    def test_not_exp_if_not_informative(self, args):
        """ Tests that if a dimension is filled, it is not in delta_a"""
        (cells, dista, feature_structures) = args
        delta_a = exponence(cells, dista, feature_structures)
        note(f"exp = {delta_a}")

        for f in feature_structures:
            vs = feature_structures[f]
            if vs <= dista :
                self.assertFalse(vs <= delta_a)

    @given(cells_dist_feats().filter(lambda args: args[0] != args[1]))
    def test_no_subsets(self, args):
        """ Tests that no element of delta_a is a subset of another"""
        (cells, dista, feature_structures) = args
        delta_a = exponence(cells, dista, feature_structures)
        note(f"exp = {delta_a}")
        self.assertFalse(any([x1 < x2 for x1, x2 in combinations(delta_a, 2)]))

    @given(cells_dist_feats())
    def test_elts_taken_from_dista(self, args):
        """ Tests that elements of delta_a are from the dista"""
        (cells, dista, feature_structures) = args
        delta_a = exponence(cells, dista, feature_structures)
        note(f"exp = {delta_a}")
        self.assertTrue(all(any(x <= y for y in dista) for x in delta_a))

    @given(cells_dist_feats().filter(lambda args: args[0] == args[1]))
    def test_empty_if_dist_is_cells(self, args):
        """ Tests that delta_a is empty if the dist_a is equal to the cells"""
        cells, dista, feature_structures = args
        delta_a = exponence(cells, dista, feature_structures)
        note(f"exp = {delta_a}")
        self.assertEqual(delta_a, set())

    @given(cells_dist_feats())
    def test_supersets_in_dista(self, args):
        """ Tests delta_a describe only the dist, no other cells

        All cells that are supersets of an $x$
        in $\delta_a$ are in the distribution of $a$
        """
        cells, dista, feature_structures = args
        delta_a = exponence(cells, dista, feature_structures)
        note(f"exp = {delta_a}")
        for z in cells:
            for x in delta_a:
                if z >= x:
                    self.assertTrue(any(y <= z for y in dista))


if __name__ == '__main__':
    unittest.main()
