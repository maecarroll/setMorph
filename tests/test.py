#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from setmorph import parse_cell
import pandas as pd

class testSetMorph(unittest.TestCase):
    def test_parse_cell(self):
        test_data = pd.DataFrame.from_dict({"cell": [
            "prs.1.sg", "prs.2.sg", "pst.1.sg"]},
            orient="columns")

        cells = parse_cell(test_data)
        expected = [{"prs", "1", "sg"},
                    {"prs", "2", "sg"},
                    {"pst", "1", "sg"},
                    ]

        self.assertListEqual(cells, expected)


if __name__ == '__main__':
    unittest.main()
