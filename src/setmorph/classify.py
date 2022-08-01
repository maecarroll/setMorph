#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .setmorph import exponence
from itertools import chain, product
import pandas as pd
from typing import NamedTuple

class Formative(NamedTuple):
    tier: str
    slot: str
    formative: str

    def __repr__(self):
        return f"<{self.tier}_{self.slot}_{self.formative}>"


def find_exponents(df, features):
    """ Returns all the exponence descriptions for an entire lexicon

    Args:
        df: a lexicon

    Returns:
        a pd.DataFrame associating each quadruple of (lexeme, tier, slot, formative)
            to a set of fv combinations it is an exponent of,
             the number of cells it occurs in,
            and its full distribution.
    """

    def exponence_word(paradigm):
        cells = set(paradigm.celllist)
        f_values = set(chain(*cells))
        features_w = {f: {v for v in features[f] if v <= f_values}
                      for f in features}
        groups = paradigm.groupby(["tier", "slot", "formative"])
        transforms = {"celllist": [lambda d: exponence(cells, set(d), features_w),
                                   len, set]}
        return groups.agg(transforms)

    result = df.groupby("lexeme").apply(exponence_word)
    result.columns = ["exponence", "# of cells", "dist_a"]
    return result.reset_index()


def classify_simple(df):
    """ Classifies all formatives in a lexicon (dataframe) as simple exponence or not.

    Modifies exps in place.
    """

    def simple(descr):
        """ Answers the question: is this description simple ?

        Args:
            descr (set): the minimal description of the distribution of a formative.

        Returns (str):
            "yes" if the description if simple, "no" if it is not, or "invariant" if the
            formative is present in all forms of the lexeme.
        """
        length = len(descr)
        return None if length == 0 else True if length == 1 else False

    df["simple"] = df["exponence"].apply(simple)


def classify_cumulation(df, max_dims):
    """ Classify formatives according to exponent cumulation

    Args:
        df (pd.DataFrame): DataFrame of formatives & their exponential value
        max_dims (int): Maximum number of dimensions in paradigms.

    Returns:
        None -- modifies the exps in place, adding the columns:

        - 'cumulative cells', a set of cumulative values
        - 'longest cumulation', the maximum number of dimensions in cumulative values
        - '% cells cumulative' the ratio of cells with cumulation for this f_row,
            compared to the number of cells in which the f_row occurs
        - '% dimensions cumulation', the ratio of longest cumulation, compared to
            the maximum number of dimensions.
    """

    def cumulation_measures(f_row):
        """ Calculate measures of exponent cumulation for a formative.

        Args:
            f_row (pd.Series): a row representing a formative.
        """
        c_vals = set(filter(lambda x: len(x) > 1, f_row["exponence"]))
        c_cells = set(y for x, y in product(c_vals, f_row.dist_a) if y <= x)
        max_vals = len(max(c_vals)) if c_vals else 0
        return pd.Series({'cumulative cells': c_vals,
                          'longest cumulation': max_vals,
                          '% cells cumulative': len(c_cells) / len(f_row.dist_a) * 100,
                          '% dimensions cumulation': max_vals / max_dims * 100})

    new_cols = ['cumulative cells', 'longest cumulation',
                '% cells cumulative', '% dimensions cumulation']
    df[new_cols] = df.apply(cumulation_measures, axis=1)


def classify_syn(df):
    """Classifies all formatives in a lexicon (dataframe) regarding syncretism.

    """
    df["# sets minimally required"] = df["exponence"].fillna("").apply(len)


def values_per_word(df, exps):  # TODO: to test
    # Build a dist_a dict: lexeme, formative, slot, tier => dist
    vals_dist = {tuple(r[["lexeme", "slot", "formative", "tier"]]): r.exponence for i, r in
                 exps[["lexeme", "slot", "formative", "tier", "exponence"]].iterrows()}

    # For each formative, add the subset of values from the cells that are exponential
    def exponential_vals(row):
        cell = row.celllist
        exp = vals_dist[(row.lexeme, row.slot, row.formative, row.tier)]
        res = tuple({*chain(*{vs for vs in exp if vs <= cell})})
        return res

    df["value"] = df.apply(exponential_vals, axis=1)


    def gather_formatives(occs):
        form_cols = ["tier", "slot", "formative"]
        if occs.shape[0] == 0:
            return None
        first = occs.iloc[0, :]
        forms = occs[form_cols].to_records(index=False)
        return pd.Series({"lexeme": first["lexeme"],
                          "value": first["value"],
                          "form": first["form"],
                          "cell": first["cell"],
                          "formative": tuple(sorted({Formative(*f) for f in forms})),
                          })

    # List exponential values expressed as separate rows
    values = df.explode("value")

    # For each value in a separate word, build a list of formatives
    values = values.groupby(["lexeme", "cell", "value", "form"],
                            as_index=False,
                            group_keys=True).apply(gather_formatives)\
        .dropna()\
        .reset_index(drop=True)
    return values


def values_table(values_words):
    """

    Args:
        values_words:

    Returns:

    """
    values = values_words.sort_values("lexeme", axis=0)
    values = values.groupby(["lexeme", "value"], as_index=False)
    values = values.agg({"formative": set})
    values["formatives_by_word"] = values.formative
    values["formative"] = values.formative.apply(lambda f: set(chain(*f)))
    return values


def count_formatives(values_words):
    """ Adds in place a count for formatives.

    Applied on values per word, this lets one detect unique exponence.
    Applied on values tables, this lets one detect verbose exponence.

    Args:
        values_words:

    Returns:

    """
    values_words["# formatives"] = values_words.formative.apply(len)


def classify_allomorphy(df, values):
    """

    Args:
        exps:

    Returns:

    """

    # Build a dict of: lexeme, val => number of words with this value
    def count_w(group):
        return group[["lexeme", "cell", "form"]].drop_duplicates().shape[0]

    val_counts = df.explode("celllist") \
        .groupby(["lexeme", "celllist"]) \
        .apply(count_w).to_dict()

    values["# allomorph sets"] = values["formatives_by_word"].apply(len)
    values["# words"] = values.apply(lambda r: val_counts[(r.lexeme, r.value)],
                                     axis=1)
    values["% allomorph to words"] = (values["# allomorph sets"] / values[
        "# words"]) * 100
