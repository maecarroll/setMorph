import pandas as pd
from itertools import combinations, chain, product
from collections import Counter


def read_paradigms(path):
    """ Read a data file.

    Args:
        path (str): path to a table of segmented formatives.

    Returns:
        a pd.Dataframe of segmented formatives. Cells are parsed into lists of frozensets.
    """
    df = pd.read_csv(path)
    df.loc[:, 'celllist'] = df.cell.str.split(".").apply(frozenset).tolist()
    return df


def shortest(cells, dista):
    """ Compute the shortest description $\delta$ of a formative's distribution.

    Args:
        cells (set): the set of cells in which a lexeme occurs
        dista (set): the set of cells in which the formative occurs

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
    for cell in dista:
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
                s = frozenset(s)
                if s in delta | subsets:
                    found = True
                else:
                    span_word = set(filter(lambda c: s < c, cells))
                    span_dista = {c for c in dista if s < c}
                    is_subset = any(span_word <= s for s in spans)
                    if span_word == span_dista and not is_subset:
                        subsets.add(s)
                        found = True
                        spans.add(frozenset(span_dista))
            i += 1
        if found:
            delta |= subsets
        else:
            delta.add(cell)

    return delta


def simple(descr):
    """ Answers the question: is this description simple ?

    Args:
        descr (set): the minimal description of the distribution of a formative.

    Returns (str):
        "yes" if the description if simple, "no" if it is not, or "invariant" if the
        formative is present in all forms of the lexeme.
    """
    l = len(descr)
    return "invariant" if l == 0 else "yes" if l == 1 else "no"


def cumulative(descr):
    """ Returns the subset of values in descr which are cumulative.

    Args:
        descr (set): the minimal description of the distribution of a formative.

    Returns:
        a set of cumulative values.
    """
    return set(filter(lambda x: len(x) > 1, descr))


def cumulative_cells(cumul_values, dist_a):
    """ Returns the cells which contain cumulative values

    Args:
        cumul_values (set): a set of cumulative values
        dist_a (set): a set of cells (distribution of a formative)

    Returns:
        a set of cells in which there are cumulative values
    """
    return set(y for x, y in product(cumul_values, dist_a) if y <= x)


def cumulation_measures(formative, max_dims):
    """ Calculate measures of exponent cumulation

    Args:
        formative (pd.Series): a row representing a formative, with its minimal description.
        max_dims (int): Maximum number of dimensions in paradigms.

    Returns:
        the formative series, augmented with:
            - 'cumulative cells', a set of cumulative values
            - 'longest cumulation', the maximum number of dimensions in cumulative values
            - '% cells cumulative' the ratio of cells with cumulation for this formative,
                compared to the number of cells in which the formative occurs
            - '% dimensions cumulation', the ratio of longest cumulation, compared to
                the maximum number of dimensions.
    """
    cells = formative["minimal description"]
    c_vals = cumulative(cells)
    c_cells = cumulative_cells(c_vals, formative.dist_a)
    max_vals = len(max(c_vals)) if c_vals else 0
    formative['cumulative cells'] = c_vals
    formative['longest cumulation'] = max_vals
    formative['% cells cumulative'] = len(c_cells) / len(formative.dist_a) * 100
    formative['% dimensions cumulation'] = max_vals / max_dims * 100
    return formative
