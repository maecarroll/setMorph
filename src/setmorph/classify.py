#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setmorph import shortest, simple, cumulation_measures
from operator import itemgetter
from itertools import groupby, chain


def delta_lexicon(df):
    """ Returns all the shortest descriptions for an entire lexicon

    Args:
        df: a lexicon

    Returns:
        a pd.DataFrame associating each quadruple of (lexeme, tier, slot, formative)
            to its minimal description, the number of cells it occurs in,
            and its full distribution.
    """

    def short_descr_word(paradigm):
        cells = set(paradigm.celllist)
        groups = paradigm.groupby(["tier", "slot", "formative"])
        transforms = {"celllist": [lambda d: shortest(cells, set(d)), len, set]}
        return groups.agg(transforms)

    result = df.groupby("lexeme").apply(short_descr_word)
    result.columns = ["minimal description", "# of cells", "dist_a"]
    return result.reset_index()


def classify_simple(df):
    """ Classifies all formatives in a lexicon (dataframe) as simple exponence or not.
    """
    df = df.copy(deep=True)
    df["simple?"] = df["minimal description"].apply(simple)
    return df


def classify_cumulation(df, max_dims):
    df = df.copy(deep=True)
    return df.apply(lambda form: cumulation_measures(form, max_dims), axis=1)


def classify_syn(df):
    """Classifies all formatives in a lexicon (dataframe) with regards to
    syncretism.

    """
    df = df.copy(deep=True)
    df["# sets minimally required"] = df["minimal description"].fillna("").apply(len)
    return df.reset_index()


def classify_unique(df):
    """

    Args:
        df:

    Returns:

    """
    df = df.copy(deep=True)

    # Get one row for each formative,  value in minimal description
    df["value"] = df["minimal description"].apply(lambda x: list(chain(*x)))
    df = df.explode("value")

    # Keep only the four columns we are interested in,
    # Remove duplicates to keep a single occurence per formative
    df = df.loc[:, ["lexeme", "value", "slot", "formative"]].drop_duplicates()

    # Make groups with the same lexeme & value
    groups = df.groupby(["lexeme", "value"])

    # Filter to return only the groups with a single formative
    return groups.filter(lambda g: g.shape[0] == 1)


def classify_allomorphy(df):
    """Classifies all values in a lexicon (dataframe) with regards to
    uniqueness.

    Crucially, treats formatives which have the same distribution with
    regard to a value as a group (to exclude ME.)

    This excludes all ME but should it treat all VE as a 'single allomorph'?
    """
    table = []
    lexemes = set(df['lexeme'])

    for lexeme in lexemes:
        dflex = df[df['lexeme'] == lexeme]
        valuelist = dflex.cell.str.split(".").apply(
            frozenset).tolist()  # gets set of values for a given lexeme but
        valueset = set()
        for cell in valuelist:
            for value in cell:
                valueset.add(value)

        # gets a list of the maximal delta for each formative
        deltatable = delta_lexicon(dflex)

        for value in valueset:  # goes through and makes a list of all for the formatives which have value in their minimum delta
            examplelist = []
            for i, delta in deltatable.dropna().iterrows():
                celllist = []
                for cell in delta['minimal description']:
                    if type(cell) != str:
                        if value in cell:
                            celllist.append(cell)
                if len(celllist) > 0:
                    examplelist.append(
                        {'lexeme': lexeme, 'value': value, 'tier': delta['tier'],
                         'slot': delta['slot'], 'form': delta['formative'],
                         'cells': celllist})
            # examplelistunique = list({str(i):i for i in examplelist}.values()) #makes list unique
            if len(examplelist) > 0:
                # THIS IS NON-UNIQUE EXPONENCE!! (use this for the others)
                # below checks to see if their distributions are identical:
                deltaset = set()
                for delta in examplelist:
                    for cell in delta['cells']:
                        deltaset.add(cell)

                # This one groups those with identical distribution with respect to a feature value:

                if len(deltaset) > 1:
                    celllist2 = []
                    examplelist.sort(key=itemgetter('cells'))

                    for key, value2 in groupby(examplelist,
                                               lambda item: item['cells']):
                        val2list = []
                        for x in value2:
                            form2 = (x['tier'], x['slot'], x['form'])
                            val2list.append(form2)
                        celllist2.append(val2list)

                    cellswithv = set(valuelist)
                    cellcount = sum(value in cell for cell in cellswithv)

                    table.append({
                        'lexeme': lexeme,
                        'value': value,
                        '# allomorphs': len(celllist2),
                        '% of allomorphs to cells containing v': len(
                            celllist2) / cellcount * 100,
                        'forms': celllist2

                    })
    return table


def classify_verbose(df):
    df = df.copy(deep=True)

    # Make separate rows for each cell
    df = df.explode("dist_a").rename(columns={"dist_a": "cell"})

    # We want a row for each cell value that is in the minimal description
    df["values"] = df.apply(lambda row: {v for v in row["minimal description"]
                                         if set(row.cell) >= v },
                            axis=1)

    # Make groups with the same value in words
    cols = ["lexeme", "cell", "values"]
    groups = df.explode("values").groupby(cols)

    # Filter to keep only groups with more than a single formative
    # given the same lexeme, cell, and value expressed
    groups = groups.filter(lambda g: g.shape[0] > 1).sort_values(cols)

    # Group again,  this time to reduce rows
    groups = groups.groupby(cols).agg({})


    return groups #.set_index(cols)


def classify_VE_old(df):
    """Classifies all values in a lexicon (dataframe) with regards to
    verbose exponence.

    """
    table = []
    lexemes = set(df['lexeme'])

    for lexeme in lexemes:
        dflex = df[df['lexeme'] == lexeme]
        # gets set of values for a given lexeme but
        valuelist = dflex.cell.str.split(".").apply(frozenset).tolist()
        valueset = set()
        for cell in valuelist:
            for value in cell:
                valueset.add(value)

        # gets a list of the maximal delta for each formative
        deltatable = delta_lexicon(dflex)

        # goes through and makes a list of all for the formatives which have value in their minimum delta
        for value in valueset:
            examplelist = []
            for i, delta in deltatable.dropna().iterrows():
                celllist = []
                for cell in delta['minimal description']:
                    if type(cell) != str:
                        if value in cell:
                            celllist.append(cell)
                if len(celllist) > 0:
                    examplelist.append(
                        {'lexeme': lexeme, 'value': value, 'tier': delta['tier'],
                         'slot': delta['slot'], 'form': delta['formative'],
                         'cells': celllist})

            if len(examplelist) > 0:  # examplelist = list of formative for value
                locations = [idx for idx, cell in enumerate(valuelist) if
                             cell > {value}]  # locations of cells containing value
                wordlist = dflex.iloc[locations]  # list of words for value
                # for each word... list all the formatives which co-occur

                for word in set(zip(wordlist.cell, wordlist.form)):
                    # print(wordlist)
                    dfword = wordlist[
                        (wordlist['cell'] == word[0]) & (wordlist['form'] == word[1])]
                    formativelist = []
                    for formative in examplelist:
                        formtuple = (
                            formative['tier'], formative['slot'], formative['form'])
                        if formtuple in zip(dfword.tier, dfword.slot, dfword.formative):
                            formativelist.append(formtuple)
                    if len(formativelist) > 1:
                        table.append({
                            'lexeme': lexeme,
                            'value': value,
                            'word': word,
                            'formatives': (formativelist),
                            '# formatives': len(formativelist)
                        })
    return table
