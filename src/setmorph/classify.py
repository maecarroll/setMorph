#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setmorph import exponence, simple, cumulation_measures
from operator import itemgetter
from itertools import groupby, chain
import pandas as pd


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
    """
    df = df.copy(deep=True)
    df["simple?"] = df["exponence"].apply(simple)
    return df


def classify_cumulation(df, max_dims):
    df = df.copy(deep=True)
    return df.apply(lambda form: cumulation_measures(form, max_dims), axis=1)


def classify_syn(df):
    """Classifies all formatives in a lexicon (dataframe) with regards to
    syncretism.

    """
    df = df.copy(deep=True)
    df["# sets minimally required"] = df["exponence"].fillna("").apply(len)
    return df.reset_index()


def classify_unique(df):
    """

    Args:
        df:

    Returns:

    """
    df = df.copy(deep=True)

    # Get one row for each value in exponence
    df["value"] = df["exponence"].apply(lambda x: list(chain(*x)))
    df = df.explode("value")

    # Keep only the four columns we are interested in,
    # Remove duplicates to keep a single occurence per formative
    df = df.loc[:, ["lexeme", "value", "slot", "formative"]].drop_duplicates()

    # Make groups with the same lexeme & value
    groups = df.groupby(["lexeme", "value"])

    # Filter to return only the groups with a single formative
    return groups.filter(lambda g: g.shape[0] == 1)


def classify_allomorphy(df):
    """

    Args:
        df:

    Returns:

    """

    # Duplicate rows to have separate rows for each cell in the distribution
    per_cell = df.explode("dist_a")
    per_cell["cell"] = per_cell.dist_a.apply(frozenset)

    # lexeme, val => number of cells with this value
    per_cell["all vals"] = per_cell.cell
    cell_counts = per_cell.explode("all vals") \
        .reset_index(drop=False) \
        .groupby(["lexeme", "all vals"]) \
        .agg({"cell": "count"}) \
        .cell \
        .to_dict()

    #  List exponential values expressed as separate rows
    per_cell["vals"] = per_cell.apply(lambda r: set(chain(*r.exponence)) & set(r.cell),
                                      axis=1)
    per_cell = per_cell.explode("vals")

    # Group formatives per word
    per_cell = per_cell.groupby(["lexeme", "cell", "vals"]).agg(
        {"dist_a": lambda x: x.iloc[0],
         "slot": tuple,
         "tier": tuple,
         "formative": tuple})

    # Count number of diff formatives across words per value
    per_cell = per_cell.groupby(["lexeme", "vals"]).agg({"formative": ("count", set)})
    # Flatten multi-indexes in columns & index
    per_cell.columns = [' '.join(col).strip() for col in per_cell.columns.values]
    per_cell = per_cell.reset_index(drop=False)

    # Compute the ratio of allomorphs to cells with the value
    total_cell = per_cell.apply(lambda x: cell_counts[(x.lexeme, x.vals)], axis=1)
    ratio_col = "% allomorphs to cells containing v"
    per_cell[ratio_col] = per_cell["formative count"] / total_cell

    return per_cell


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
