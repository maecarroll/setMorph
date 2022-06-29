#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from setmorph import exponence
from pathlib import Path
from hypothesis import given, note
from itertools import combinations, chain
from strategies import cells_dist_feats

here = Path(__file__)


class testAllomorphy(unittest.TestCase):

    @given()
    def test_x(self, args):

if __name__ == '__main__':
    unittest.main()
