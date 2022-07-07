#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames
import pandas as pd

# features or values
@st.composite
def word(draw, min_size=0, max_size=None):
    C = st.sampled_from("bcdfghjklmnpqrstvwxyz")
    V = st.sampled_from("aeiou")
    if max_size is None:
        max_size = draw(st.integers(min_size=0))
    syllables = [draw(V) if i % 2 == 0 else draw(C) for i in range(min_size, max_size)]
    return "".join(syllables)


abbr = st.text("bcdfghjklmnpqrstvwxyzaeiou", min_size=1, max_size=3)


@st.composite
def feature_structures(draw):
    """ Strategy to create synthetic feature structures

    Featurestructures look like:
        {   feat: {frozenset({val1}), frozenset({val2})},
            feat2: {... } ...
        }

    We want all abbreviations, be them features or values, to be unique.

    Args:
        draw:

    Returns:

    """
    # Pick a vocabulary of abbreviations (a unique set of abbreviations)
    abbrs = draw(st.sets(abbr, min_size=5, max_size=300))
    l = len(abbrs)

    # Select a set of abbreviations to serve as features
    #  and remove them from the vocabulary
    features = draw(st.sets(st.sampled_from(sorted(abbrs)), min_size=1, max_size=l // 2))
    abbrs = abbrs - features

    # Build the dictionnary of feature => { frozenset({value}), ... }
    # Sample some abbreviations to serve as values for each feature,
    # removing them from the vocabulary each time
    # until either features or vocabulary are exhausted
    fs = {}
    for f in features:
        values = draw(st.sets(st.sampled_from(sorted(abbrs)), min_size=1, max_size=20))
        abbrs = abbrs - values
        fs[f.upper()] = {frozenset({v}) for v in values}
        if len(abbrs) == 0:
            break
    return fs


@st.composite
def cell(draw, fs):
    """ Strategy to create synthetic cells

    cells are frozensets of values,
     such that each value is taken from a distinct feature.

    Args:
        draw:
        fs: a feature_structure

    Returns:

    """
    # select a sample of features
    features = draw(st.sets(st.sampled_from(sorted(fs)), min_size=1))
    # pick one value in each feature
    values = [draw(st.sampled_from(sorted(fs[f]))) for f in features]
    return frozenset.union(*values)


@st.composite
def cell_feats(draw):
    features = draw(feature_structures())
    cells = draw(st.sets(cell(features), min_size=2, max_size=600))
    return cells, features


@st.composite
def cells_dist_feats(draw):
    """ Strategy to create synthetic distributions, for specific dists and cells.

    This is a random abstract paradigm generator,
    with a distribution for a single formative.

    Args:
        draw:

    Returns:

    """
    cells, features = draw(cell_feats())
    dist = draw(st.sets(st.sampled_from(sorted(cells)), min_size=1, max_size=600))
    return (cells, dist, features)


@st.composite
def exponents_df(draw):
    """ This represents a dataframe of segmented exponents

    Returns:
        a strategy to generate false exponent dataframes
    """
    # select a set of lexemes
    lexemes = draw(st.sets(word(min_size=3, max_size=6), min_size=1, max_size=10))

    # Choose a paradigm structure
    cells, features = draw(cell_feats())
    l = len(cells)

    rows = []
    for lex in lexemes:
        # pick a subset of cells, at least 2, at most all cells
        lex_cells = draw(st.sets(st.sampled_from(sorted(cells)), min_size=2, max_size=l))
        l_paradigm = len(lex_cells)

        # create a set of exponents
        # At least one,
        # At most 3 distinct exponents per cell
        exps = draw(st.sets(word(min_size=1, max_size=4),
                            min_size=1,
                            max_size=l_paradigm * 3))

        l_exps = len(exps)
        # assign exponents to tiers -- keeping this simple
        tiers = draw(st.lists(st.sampled_from(["segmental", "tone"]),
                              min_size=l_exps, max_size=l_exps))
        # Set of exponents, plus the stem
        tiered_exps = [("segmental", lex[:-1])] + list(zip(tiers, exps))

        for c in lex_cells:
            # Pick between 1 and 3 forms (overabundance)
            form_count = draw(st.integers(min_value=1, max_value=3))

            for _ in range(form_count):
                # Build a single form as a set of formatives
                formatives = list(draw(st.sets(st.sampled_from(tiered_exps),
                                               min_size=1,
                                               max_size=3)))
                form = " ".join("".join([f for t, f in formatives]))
                for i, (t, f) in enumerate(formatives):
                    rows.append({"lexeme": lex,
                                 "cell": ".".join(c),
                                 "form": form,
                                 "tier": t,
                                 "slot": i,
                                 "formative": f,
                                 "celllist": c
                                 })
    df = pd.DataFrame(rows)
    return df, features
