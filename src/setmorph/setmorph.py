import logging
import pandas as pd
from itertools import combinations, chain, product
from collections import Counter
from typing import NamedTuple
from tqdm import tqdm
tqdm.pandas()

class Formative(NamedTuple):
    """A formative is a triple

    of a phonological tier, a slot in a paradigm,
    and a specific form.

    """
    tier: str
    slot: str
    formative: str

    def __repr__(self):
        return f"<{self.tier}_{self.slot}_{self.formative}>"


def check_cell_structure(cells):
    """ Checks that the cells are not malformed.

    - Cells must not be subsets of other cells.

    Args:
        cell_series: a set of cells

    Returns:
        None if everything is alright, otherwise throws an exception.

    """
    incl = []
    cells = sorted(cells, key=len)
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
    """ Reads a features file.

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
    """ Reads a paradigms file.

    Args:
        path (str): path to a table of segmented formatives.

    Returns:
        a pd.Dataframe of segmented formatives. Cells are parsed into lists of frozensets.
    """
    checked = set()
    def check_lexeme_cells(group):
        cells = frozenset(group.cell.unique())
        if cells not in checked:
            checked.add(cells)
            check_cell_structure(cells)
    df = pd.read_csv(path)
    df.loc[:, 'cell'] = df.cell.str.split(".").apply(frozenset)
    df.groupby("lexeme").apply(check_lexeme_cells)
    return df


def exponence(cells, dista, feature_structure):
    """ Calculates the set of values a formative is the exponent of, from its distribution.

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


def get_real_per_word(df, reals):
    """ Associates (value, word) pairs to their realisations.

    This produces a table where rows represent values in words.
    The column 'real_w' represents the realization of this value
    in this word.

    Args:
        df: a table of paradigms
        reals: a  table for realizations

    Returns:
        a table for realizations in words
    """

    def gather_formatives(occs):
        form_cols = ["tier", "slot", "formative"]
        if occs.shape[0] == 0:
            return None
        first = occs.iloc[0, :]
        forms = occs[form_cols].to_records(index=False)
        return pd.Series({"lexeme": first["lexeme"],
                          "phon_form": first["phon_form"],
                          "cell": first["cell"],
                          "wordform": frozenset(sorted({Formative(*f) for f in forms})),
                          })

    cols = ['lexeme', 'phon_form', 'cell', 'wordform', 'vals', 'real_w', "|real_w|"]

    if reals.shape[0] == 0:
        res = pd.DataFrame(columns=cols )
        return res

    # For each value in a separate word,
    # build a list of all of the formatives in this word
    real_w = df.groupby(["lexeme", "cell", "phon_form"],
                            as_index=False,
                            group_keys=True).progress_apply(gather_formatives)\
        .dropna()\
        .reset_index(drop=True)
    real_w["vals"] = real_w["cell"]
    real_w = real_w.explode("vals")

    # Calculate real_w: the formatives in this word which express the value
    real_w = pd.merge(reals, real_w,
                      left_index=True, right_on=["vals", "lexeme"])
    real_w["real_w"] = real_w.progress_apply(lambda row: row["real"] & row["wordform"], axis=1)
    real_w["|real_w|"] = real_w["real_w"].apply(len)
    return real_w[cols]


def get_reals(exponents):
    """ Calculates the *real* variable for each value in a paradigm.

    Args:
        exponents: the exponents table

    Returns:
        a table mapping of each value to a set of


    """
    def gather_formatives(occs):
        form_cols = ["tier", "slot", "formative"]
        if occs.shape[0] == 0:
            return None
        forms = occs[form_cols].to_records(index=False)
        reals = frozenset({Formative(*f) for f in forms})
        return pd.Series({"real": reals,
                          "|real|": len(reals)})

    res = exponents.explode("vals")\
                    .groupby(["vals", "lexeme"])\
                    .progress_apply(gather_formatives)

    if res.shape[0] == 0:
        res = pd.DataFrame(columns=['vals', 'lexeme',
                                    "real", "|real|"])
        res.set_index(['vals', 'lexeme'], inplace=True)
    return res

def get_exponents(df, features):
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
        cells = set(paradigm.cell)
        f_values = set(chain(*cells))
        features_w = {f: {v for v in features[f] if v <= f_values}
                      for f in features}
        groups = paradigm.groupby(["tier", "slot", "formative"])
        transforms = {"cell": [frozenset, lambda d: exponence(cells, set(d), features_w)],
                      "full_slot": [lambda x: set(tuple(x))]}
        res = groups.agg(transforms)
        res.columns = ["dist", "exponence", "full_slot"]
        res["vals"] = res.exponence.apply(lambda x: set(chain(*x)))
        res["|vals|"] = res["vals"].apply(len)
        res["|exp|"] = res["exponence"].apply(len)
        return res

    def merge_same_dist(exps):
        first = exps.iloc[0,:]
        if exps.shape[0] == 1:
            return first
        chars = set(chain(*exps["formative"]))
        first["formative"] = "".join(c for c in first["full_slot"].pop() if c in chars)
        first["tier"] = "/".join(exps.tier.sort_values())
        return first
    result = df.groupby(["lexeme"]).progress_apply(exponence_word).reset_index()
    result = result.groupby(["lexeme", "slot", "dist"], as_index=False).apply(merge_same_dist)
    result.drop("full_slot", inplace=True, axis=1)
    return result


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

        - 'cumulative', a set of cumulative f-values combinations
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
        c_cells = set(y for x, y in product(c_vals, f_row.dist) if x <= y)
        max_dims = max(len(c) for c in f_row.dist)
        max_vals = len(max(c_vals)) if c_vals else 0
        return pd.Series({'cumulative': c_vals,
                          'cumulative_cells': c_cells,
                          'max_cumulation': max_vals,
                          'maximum_possible_cumulation': max_dims})

    new_cols = ['cumulative', 'cumulative_cells',
                'max_cumulation',
                'maximum_possible_cumulation']
    df[new_cols] = df.progress_apply(cumulation_formative, axis=1)