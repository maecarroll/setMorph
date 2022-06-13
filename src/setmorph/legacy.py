#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .setmorph import shortest
from itertools import chain, combinations, product
import pandas as pd


def feat_set(sig, feat_df):
    feat_set = set()
    for value in feat_df[feat_df['feature'] == sig]['value']:
        feat_set.add(value)
    return feat_set

def parse_formatives(df):
    formativelist = list(zip(df.tier, df.slot, df.formative))
    df['formtuples'] = formativelist
    wordformset = []
    for wordform in df['form']:
        wordform_asset = frozenset(df['formtuples'][df['form'] == wordform])
        wordformset.append(wordform_asset)
    return wordformset

def parse_formatives_lex(df):
    formativelist = list(zip(df.lexeme, df.tier, df.slot, df.formative))
    df['formtuples'] = formativelist
    wordformset = []
    for wordform in df['form']:
        wordform_asset = frozenset(df['formtuples'][df['form'] == wordform])
        wordformset.append(wordform_asset)
    return wordformset

def real(val, lex, df):  ## Assumes that all feature values are unique
    val = {val}
    real = df[(df['lexeme'] == lex) &
              (df['celllist'] & val)]
    return real

def shortest_old(cells, dista):
    """ Shortest description $\delta$ of a formative's distribution.

    Args:
        cells: the set of cells in which a lexeme occurs
        dista: the set of cells in which the formative occurs

    Returns:
        delta: a description of the distribution which is as short as possible.

    """
    if dista == cells: return None
    fullvalueset = set(chain(*cells))  # all possible feature value combinations

    bigdeltapossible = {frozenset(d) for i in range(len(max(cells))) for d in
                        combinations(fullvalueset, i + 1)}
    # set of all possible descriptions (feature value combinations)

    # the above two for loops can reduced to a single one that reduces the cell sets down rather than combining and filtering

    # filter those which are in the lexeme:
    bigdeltaword = set(filter(lambda d: any(d <= c for c in cells), bigdeltapossible))

    bigdelta = set()
    # filter those which are accurate descriptions of dista:
    # It is an accurate description if,
    # All cells in dista
    for delta in bigdeltaword:
        matchdista = set()
        matchword = set()
        for cell in dista:
            if delta <= cell:
                matchdista.add(cell)
        for cell in cells:
            if delta <= cell:
                matchword.add(cell)
        if matchdista == matchword:
            bigdelta.add(delta)

    # filter those which are supersets of the other:
    combo = combinations(bigdelta, 2)
    for a, b in combo:
        if a < b:
            if b in bigdelta:
                bigdelta.remove(b)
        elif b < a:
            if a in bigdelta:
                bigdelta.remove(a)

    # check to see if one implies the other? i.e. cells which exist as a subset of another
    for a, b in combinations(bigdelta, 2):
        matcha = set()
        matchb = set()
        for cell in cells:
            cell = frozenset(cell)
            if a <= cell:
                matcha.add(cell)
            if b <= cell:
                matchb.add(cell)
        if matcha < matchb:
            bigdelta.remove(a)
        elif matchb < matcha:
            bigdelta.remove(b)
        elif matcha == matchb:
            print('Redundant feature structure. Features ' + str(a) + ' ' + str(
                b) + ' are equivilent.')

    return bigdelta


def dist(a, df):
    """

    Args:
        a:
        df:

    Returns:

    """
    cols = ["tier", "slot", "formative", "lexeme"]
    if type(df) is not pd.Series:
        a = pd.Series(dict(zip(cols, a)))

    dist = df[(df[cols] == a).all(axis=1)]
    return set(dist.celllist)

def shortest_description(tier, pos, form, lex, df):
    df = df[df['lexeme'] == lex]  # <-- weird, seems to be done outside too
    dista = dist((tier, pos, form, lex), df)
    return shortest(set(df.celllist), dista)

def classify_ve_old(df):
    """Classifies all words in a lexicon (dataframe) with regards to verbose
    exponence. This is an older implementation based on words rather than
    values and it only searches for pairs of formatives. This is closer to the
    analysis found in Caroll (2022).
    """
    table = []
    wordlist = set(zip(df.lexeme, df.cell, df.form))
    for word in wordlist:
        df2 = df[
            (df['lexeme'] == word[0]) & (df['cell'] == word[1]) & (df['form'] == word[2])]
        formativelist = set(zip(df2.tier, df2.slot, df2.formative))
        for a, b in combinations(formativelist, 2):
            deltaa = shortest_description(a[0], a[1], a[2], word[0], df)
            deltab = shortest_description(b[0], b[1], b[2], word[0], df)
            if type(deltaa) == set:
                if type(deltab) == set:
                    for combo in product(deltaa, deltab):
                        forma = combo[0]
                        formb = combo[1]
                        if forma & formb:
                            table.append({
                                'lexeme': word[0],
                                'cell': word[1],
                                'form': word[2],
                                'form a': a,
                                'delta a': deltaa,
                                'form b': b,
                                'delta b': deltab,
                                'verbosely expressed value': forma & formb
                            })
    return table

