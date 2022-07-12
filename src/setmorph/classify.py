#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .setmorph import exponence
from itertools import chain, product
import pandas as pd
from collections import namedtuple, defaultdict

F = namedtuple("Formative",
               ["tier", "slot", "formative"])


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
        l = len(descr)
        return None if l == 0 else True if l == 1 else False

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
    """Classifies all formatives in a lexicon (dataframe) with regards to
    syncretism.

    """
    df["# sets minimally required"] = df["exponence"].fillna("").apply(len)


def classify_unique(df):
    """

    Args:
        df:

    Returns:

    """
    df = df.copy(deep=True)

    # Get one row for each value in exponence
    df["value"] = df["exponence"].apply(lambda x: set(chain(*x)))
    df = df.explode("value")

    # Keep only the four columns we are interested in,
    # Remove duplicates to keep a single occurence per formative
    df = df.loc[:, ["lexeme", "value", "slot", "formative"]].drop_duplicates()

    # Make groups with the same lexeme & value
    groups = df.groupby(["lexeme", "value"])

    # Filter to return only the groups with a single formative
    return groups.filter(lambda g: g.shape[0] == 1)


def classify_allomorphy(exps, df):
    """

    Args:
        exps:

    Returns:

    """

    # Build a dict of: lexeme, val => number of cells with this value
    val_counts = df.explode("celllist") \
        .reset_index(drop=False) \
        .groupby(["lexeme", "celllist"]) \
        .agg({"cell": "count"}) \
        .cell \
        .to_dict()

    # Build a dist_a dict: lexeme, formative, slot, tier => dist
    vals = {tuple(r[:4]): r.exponence for i, r in
            exps[["lexeme", "slot", "formative", "tier", "exponence"]].iterrows()}

    # For each formative, add the subset of values from the cells that are exponential
    def exponential_vals(row):
        cell = row.celllist
        exp = vals[(row.lexeme, row.slot, row.formative, row.tier)]
        return tuple(chain(*{vs for vs in exp if vs <= cell}))

    df["vals"] = df.apply(exponential_vals, axis=1)


    # List exponential values expressed as separate rows
    per_val = df.explode("vals")

    def allomorphy(group):
        """ Formatives are tuples of: word, tier, slot, sounds """
        l, v = group.name

        # Dictionary of formatives to sets of words they occur in with this value
        form_to_words = defaultdict(set)

        for i, r in group.iterrows():
            f = F(r.tier, r.slot, r.formative)
            w = (r.cell, r.form)
            form_to_words[f].add(w)

        # Dictionary of sets of words to sets of formatives.
        words_to_form = defaultdict(set)

        for f, w in form_to_words.items():
            words_to_form[frozenset(w)].add(f)

        formative_sets = {tuple(sorted(fset)) for fset in words_to_form.values()}

        cell_count = val_counts[(l, v)]

        infos = {"allomorph set": formative_sets,
                 "allomorph set count": len(formative_sets),
                 "cells with v": cell_count,
                 "% allomorphs to cells containing v": (
                                                                   len(formative_sets) / cell_count) * 100,
                 }
        return pd.Series(infos)

    per_val = per_val.groupby(["lexeme", "vals"]).apply(allomorphy)

    if per_val.shape[0] > 0:
        return per_val[(per_val["allomorph set count"] > 1)].reset_index(drop=False)

    return pd.DataFrame(
        columns=['lexeme', 'vals', 'allomorph set', 'allomorph set count', 'cells with v',
                 '% allomorphs to cells containing v'])


def classify_verbose(df):
    def verbose_summary(occs):
        if occs.shape[0] == 1:
            return None
        form_cols = ["tier", "slot", "formative"]
        first = occs.iloc[0, :]
        forms = occs[form_cols].to_records(index=False)
        return pd.Series({"lexeme": first["lexeme"],
                          "value": first["values"],
                          "cell": first["cell"],
                          "formatives": [tuple(f) for f in forms],
                          })

    df = df.copy(deep=True)

    # Make separate rows for each cell
    df = df.explode("dist_a").rename(columns={"dist_a": "cell"})

    # Make a row for each cell value that is in the minimal description
    # These are the values of the cell which are expressed by formatives
    df["values"] = df.apply(lambda row: set(chain(*{v for v in row["exponence"]
                                                    if set(row.cell) >= v})),
                            axis=1)

    # Make a row for each value, then
    # group formatives which occur in the same words for the same value
    res = df.explode("values").groupby(["lexeme", "cell", "values"],
                                       as_index=False,
                                       group_keys=True)

    # Keep groups with more than a single formative,
    # reshape to have one formative per row
    res = res.apply(verbose_summary).dropna()

    # Count formatives
    res["# formatives"] = res.formatives.apply(len)

    return res[["lexeme", "cell", "value", "formatives", "# formatives"]]
