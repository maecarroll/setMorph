#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import chain, product
import pandas as pd


def classify_allomorphy(reals, real_w):
    """

    Args:
        real_w:

    Returns:

    """
    allom = real_w.groupby(["lexeme", "vals"]).agg({"real_w": set})
    allom.columns = ["allomorphic_sets"]
    return pd.merge(reals, allom, left_index=True, right_index=True)


def classify_cumulation(df):
    """ Classify formatives according to exponent cumulation

    Args:
        df (pd.DataFrame): DataFrame of formatives & their exponential value

    Returns:
        None -- modifies the exps in place, adding the columns:

        - 'cumulative_vals', a set of cumulative values
        - 'cumulative_cells', a set of cells in which the formative is cumulative
        - 'maximum_possible_cumulation', the length of the longest possible cumulation,
            for this distribution. That is to say, given a dist, the cardinality of the cell
            involving the highest number of dimensions
        - 'max_cumulation': the size of the cumulative value involving the highest number
            of dimensions.
    """

    def cumulation_formative(f_row):
        """ Measures cumulation for a formative.

        Args:
            f_row (pd.Series): a row representing a formative.
        """
        c_vals = set(filter(lambda x: len(x) > 1, f_row["exponence"]))
        c_cells = set(y for x, y in product(c_vals, f_row.dist) if y <= x)
        max_dims = max(len(c) for c in f_row.dist)
        max_vals = len(max(c_vals)) if c_vals else 0
        return pd.Series({'cumulative_vals': c_vals,
                          'cumulative_cells': c_cells,
                          'max_cumulation': max_vals,
                          'maximum_possible_cumulation': max_dims})

    new_cols = ['cumulative_vals', 'cumulative_cells',
                'max_cumulation',
                'maximum_possible_cumulation']
    df[new_cols] = df.apply(cumulation_formative, axis=1)


def count_elts(df, column, name):
    """ Adds in place length for some column.

    Args:
        values_words:

    Returns:

    """
    df[name] = df[column].fillna("").apply(len)
