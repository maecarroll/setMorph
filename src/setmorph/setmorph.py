import pandas as pd
from itertools import combinations, chain
from collections import Counter


def read_features(path):
    """ Read a
    Args:
        path (str): path to a table of features.

    Returns:
        A dict of features to sets of fozensets containing each a single  value.
        values are given as frozensets to facilitate set operations.
    """
    df = pd.read_csv(path)
    df["value_id"] = df["value_id"].apply(lambda x: frozenset({x}))
    features_to_values = df.groupby("feature").agg(set).to_dict()["value_id"]
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
    return df


def exponence(cells, dista, feature_structure):
    """ Compute the exponence description a formative's distribution.

    Args:
        cells (set): the set of cells in which a lexeme occurs
        dista (set): the set of cells in which the formative occurs
        feature_structure (dict): A mapping of feature names (eg. "tense")
            to sets of values (eg. {"pst", "prs", "fut"}

    Returns:
        delta (set): a description of the distribution which is as short as possible.
            If the distribution is identical to the set of cells, then the minimal
            description is empty.
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
        l = len(cell)
        i = 1

        # We explore subsets of the cell of increasing length
        # For the subset relation check to work, we need to
        #  look first at features present in more cells
        while not found and i < l:
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