#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from hypothesis import strategies as st

### Setup strategies for synthetic data

# features or values
abbr = st.text("abcdefghijklmnopqrstuvwxyz", min_size=2, max_size=6)

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
    features = draw(st.sets(st.sampled_from(sorted(abbrs)), min_size=1, max_size=l//2))
    abbrs = abbrs - features


    # Build the dictionnary of feature => { frozenset({value}), ... }
    # Sample some abbreviations to serve as values for each feature,
    # removing them from the vocabulary each time
    # until either features or vocabulary are exhausted
    fs = {}
    for f in features:
        values = draw(st.sets(st.sampled_from(sorted(abbrs)), min_size=1, max_size=20))
        abbrs = abbrs - features
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
def cells_dist_feats(draw):
    """ Strategy to create synthetic distributions, for specific dists and cells.

    Args:
        draw:

    Returns:

    """
    features = draw(feature_structures())
    cells = draw(st.sets(cell(features), min_size=2, max_size=600))
    dist = draw(st.sets(st.sampled_from(sorted(cells)), min_size=1, max_size=600))
    return (cells, dist, features)