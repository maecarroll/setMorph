import logging
import pandas as pd
from itertools import combinations, chain
from collections import Counter


def check_cell_structure(cell_series):
    """ Checks that the cells are not malformed.

    - Cells must not be subsets of other cells.

    Args:
        cell_series: the celllist column of the paradigms df.

    Returns:
        None if everything is alright, otherwise throws an exception.

    """
    incl = []
    cells = sorted(map(set, cell_series.unique()), key=len)
    for i, c1 in enumerate(cells):
        for c2 in cells[i + 1:]:
            if c1 < c2:
                incl.append(".".join(c1) + " < " + ".".join(c2))
    if incl:
        msg = "Malformed paradigm structure: " \
              "cells can not be subsets of other cells. " \
              "I found:\n" + "\n".join(incl)
        raise ValueError(msg)


def read_features(path):
    """ Read a features file.

    Args:
        path (str): path to a table of features.

    Returns:
        A dict of features to sets of fozensets containing each a single  value.
        values are given as frozensets to facilitate set operations.
    """
    df = pd.read_csv(path)
    df["value_id"] = df["value_id"].apply(lambda x: frozenset({x}))
    features_to_values = df.groupby("feature").agg(set).to_dict()["value_id"]
    for f, vs in features_to_values.items():
        if len(vs) == 1:
            v = "".join(list(vs)[0])
            logging.warning(f"The feature `{f}` has a single value `{v}`. "
                            f"No exponents of `{v}` will be possible.")
    return features_to_values


def read_paradigms(path):
    """ Read a paradigms file.

    Args:
        path (str): path to a table of segmented formatives.

    Returns:
        a pd.Dataframe of segmented formatives. Cells are parsed into lists of frozensets.
    """
    df = pd.read_csv(path)
    df.loc[:, 'celllist'] = df.cell.str.split(".").apply(frozenset).tolist()
    check_cell_structure(df.celllist)
    return df


def exponence(cells, dista, feature_structure):
    """ Calculate the set of values a formative is the exponent of, from its distribution.

    This is calculated based on the formative's distribution. The set of exponential values
    is the set of descriptions which are in all of the deltas. Deltas are alternate ways of
    writing the distribution.

    Args:
        cells (set): the set of cells in which a lexeme occurs
        dista (set): the set of cells in which the formative occurs
        feature_structure (dict): A mapping of feature names (eg. "tense")
            to sets of values (eg. {"pst", "prs", "fut"}

    Returns:
        delta (set): a set of frozensets of values, representing exponential values.
    """
    if dista == cells:
        return set()

    # We check whether each cell in dista can be written more economically
    # Using subsets of its features
    freqs = Counter(chain(*cells))
    delta = set()
    spans = set()

    dista_ordered = sorted(dista,
                           key=lambda c: min([freqs[f] for f in c]),
                           reverse=True)
    for cell in dista_ordered:
        subsets = set()
        found = False
        length = len(cell)
        i = 1

        # We explore subsets of the cell of increasing length
        # For the subset relation check to work, we need to
        #  look first at features present in more cells
        while not found and i < length:
            combos = combinations(cell, i)
            for s in sorted(combos,
                            key=freqs.__getitem__,
                            reverse=True):
                # print(f"is {s} a description ?")
                s = frozenset(s)
                if s in delta | subsets:
                    # print(f"\t already in subsets or delta")
                    found = True
                else:
                    span_word = set(filter(lambda c: s <= c, cells))
                    span_dista = {c for c in dista if s <= c}
                    is_subset = any(span_word <= s for s in spans)
                    # print(f"\t span word: {span_word}")
                    # print(f"\t span dista: {span_dista}")
                    if span_word == span_dista and not is_subset:
                        subsets.add(s)
                        found = True
                        spans.add(frozenset(span_dista))
            i += 1
        if found:
            delta |= subsets
        else:
            delta.add(cell)

    # Check and remove dimensions which are fully filled
    for f, vs in feature_structure.items():
        if vs <= delta:
            delta -= vs
    return delta
