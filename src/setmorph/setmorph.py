import itertools
import pandas as pd
from operator import itemgetter

def feat_set(sig, feat_df):
    feat_set = set()
    for value in feat_df[feat_df['feature'] == sig]['value']:
        feat_set.add(value)
    return feat_set


def parse_cell(df):
    """ Parses the cells into  a set of values.

    We assume that users provide the cells in a
    'cell' column, that values are dot separated,
    and unique across dimensions.
    This function mostly just splits on dots.

    Args:
        df (pandas.DataFrame):  A dataframe representing
            the formatives.

    Returns: a list of frozenset of features-values.
    """
    return df.cell.str.split(".").apply(frozenset).tolist()

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


def dist(tier, pos, form, lex, df):
    dist = df[(df['lexeme'] == lex) &
              (df['tier'] == tier) &
              (df['slot'] == pos) &
              (df['formative'] == form)]
    return dist


def real(val, lex, df):  ## Assumes that all feature values are unique
    cells = parse_cell(df)
    df['celllist'] = cells
    val = {val}
    real = df[(df['lexeme'] == lex) &
              (df['celllist'] & val)]
    return real


def shortest_description(tier, pos, form, lex, df):
    df = df[df['lexeme'] == lex]
    cells = parse_cell(df)
    df['celllist'] = cells
    dista = dist(tier, pos, form, lex, df)

    #remove invariant elements:
    if set(df['celllist']) == set(dista['celllist']):
        return 'invariant'

    fullvalueset = set() #all possible feature value combinations
    #gives us a set of all the values:
    for cell in cells:
        for value in cell:
            fullvalueset.add(value)

    bigdeltapossible = set() #set of all possible descriptions (feature value combinations)
    #all possible combinations of values
    for i in range(1, len(max(cells))+1):
        for delta in itertools.combinations(fullvalueset, i):
            bigdeltapossible.add(frozenset(delta))

    #the above two for loops can reduced to a single one that reduces the cell sets down rather than combining and filtering

    bigdeltaword = set()
    #filter those which are in the lexeme:
    for delta in bigdeltapossible:
        for cell in cells:
            if delta <= cell:
                bigdeltaword.add(delta)

    bigdelta = set()
    #filter those which are accurate descriptions of dista:
    for delta in bigdeltaword:
        matchdista = set()
        matchword = set()
        for cell in dista['celllist']:
            if delta <= cell:
                matchdista.add(cell)
        for cell in cells:
            if delta <= cell:
                matchword.add(cell)
        if matchdista == matchword:
            bigdelta.add(delta)


    #filter those which are subsets of the other:
    combo = itertools.combinations(bigdelta, 2)
    for a, b in combo:
        if a < b:
            if b in bigdelta:
                bigdelta.remove(b)
        elif b < a:
            if a in bigdelta:
                bigdelta.remove(a)

    #check to see if one implies the other? i.e. cells which exist as a subset of another
    for a, b in itertools.combinations(bigdelta, 2):
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
            print('Redundant feature structure. Features ' + str(a) + ' ' + str(b) + ' are equivilent.' )


    return bigdelta



def classify_simple(df):
    """Classifies all formatives in a lexicon (dataframe) as simple exponence or not.

    """
    table = []
    lexemes = set(df['lexeme'])
    for lexeme in lexemes:
        df2 = df[df['lexeme']== lexeme]
        df2['celllist'] = parse_cell(df2)
        formativelist = set(zip(df2.tier, df2.slot, df2.formative))
        for form in formativelist:
            dista = dist(form[0], form[1], form[2], lexeme, df2)
            description = shortest_description(form[0], form[1], form[2], lexeme, df2)
            if type(description) == set:
                if len(description) == 1:
                    table.append({
                    'lexeme' : lexeme ,
                    'tier' : form[0] ,
                    'slot' : form[1] ,
                    'formative' : form[2] ,
                    'simple?' : 'yes'
                 })
                else:
                    table.append({
                    'lexeme' : lexeme ,
                    'tier' : form[0] ,
                    'slot' : form[1] ,
                    'formative' : form[2] ,
                    'simple?' : 'no'
                 })
            else:
                table.append({
                    'lexeme' : lexeme ,
                    'tier' : form[0] ,
                    'slot' : form[1] ,
                    'formative' : form[2] ,
                    'simple?' : 'invariant'
                 })
    return table


def classify_cumulation(df):
    """Classifies all formatives in a lexicon (dataframe) with regards to cumulation.

    """
    table = []
    lexemes = set(df['lexeme'])
    for lexeme in lexemes:
        df2 = df[df['lexeme']== lexeme]
        df2['celllist'] = parse_cell(df2)
        formativelist = set(zip(df2.tier, df2.slot, df2.formative))
        for form in formativelist:
            dista = dist(form[0], form[1], form[2], lexeme, df2)
            description = shortest_description(form[0], form[1], form[2], lexeme, df2)
            cumulationlist = []
            for cell in description:
                if type(cell) == frozenset:
                    if len(cell) > 1:
                        cumulationlist.append(cell)
            if len(cumulationlist) > 1:
                cumulationcells = set()
                distalist = df.cell.str.split(".").apply(frozenset).tolist()
                for x in cumulationlist:
                    for y in distalist:
                        if y <= x:
                            cumulationcells.add(y)
                table.append({
                    'lexeme' : lexeme ,
                    'tier' : form[0] ,
                    'slot' : form[1] ,
                    'formative' : form[2] ,
                    'cumulative cells' : cumulationlist ,
                    'longest cumulation' : len(max(cumulationlist)),
                    '% cells cumulative' : len(cumulationcells) / len(dista) * 100,
                    '% dimensions cumulation' : len(max(cumulationlist)) / len(max(df2['celllist'])) * 100
                 })
            else:
                table.append({
                    'lexeme' : lexeme ,
                    'tier' : form[0] ,
                    'slot' : form[1] ,
                    'formative' : form[2] ,
                    'cumulative cells' : 0 ,
                    'longest cumulation' : 0,
                    '% cells cumulative' : 0,
                    '% dimensions cumulation' : 0
                 })
    return table


def classify_syn(df):
    """Classifies all formatives in a lexicon (dataframe) with regards to
    syncretism.

    """
    table = []
    lexemes = set(df['lexeme'])
    for lexeme in lexemes:
        df2 = df[df['lexeme']== lexeme]

        formativelist = set(zip(df2.tier, df2.slot, df2.formative))
        for form in formativelist:
            description = shortest_description(form[0], form[1], form[2], lexeme, df2)
            dista = dist(form[0], form[1], form[2], lexeme, df2)
            if type(description) == str:
                table.append({
                    'lexeme' : lexeme ,
                    'tier' : form[0] ,
                    'slot' : form[1] ,
                    'formative' : form[2] ,
                    'minimal description' : description ,
                    '# sets minimally required' : 0,
                    '# of cells' : len(dista) ,
                    #'% paradigm' : len(description) / len(dista)
                 })
            elif len(description) > 1:
                table.append({
                    'lexeme' : lexeme ,
                    'tier' : form[0] ,
                    'slot' : form[1] ,
                    'formative' : form[2] ,
                    'minimal description' : description ,
                    '# sets minimally required' : len(description) ,
                    '# of cells' : len(dista),
                    #'% paradigm' : len(description) / len(dista)
                 })
            else:
                table.append({
                    'lexeme' : lexeme ,
                    'tier' : form[0] ,
                    'slot' : form[1] ,
                    'formative' : form[2] ,
                    'minimal description' : description ,
                 })
    return table


def classify_ve_old(df):
    """Classifies all words in a lexicon (dataframe) with regards to verbose
    exponence. This is an older implementation based on words rather than
    values and it only searches for pairs of formatives. This is closer to the
    analysis found in Caroll (2022).
    """
    table = []
    wordlist = set(zip(df.lexeme, df.cell, df.form))
    for word in wordlist:
        df2 = df[(df['lexeme'] == word[0])  & (df['cell']==word[1]) & (df['form']==word[2])]
        formativelist = set(zip(df2.tier, df2.slot, df2.formative))
        for a,b in itertools.combinations(formativelist, 2):
            deltaa = shortest_description(a[0], a[1], a[2], word[0], df)
            deltab = shortest_description(b[0], b[1], b[2], word[0], df)
            if type(deltaa) == set:
                if type(deltab) == set:
                    for combo in itertools.product(deltaa, deltab):
                        forma = combo[0]
                        formb = combo[1]
                        if forma & formb:
                            table.append({
                                'lexeme': word[0],
                                'cell' : word[1],
                                'form' : word[2],
                                'form a' : a,
                                'delta a' : deltaa,
                                'form b' : b,
                                'delta b' : deltab,
                                'verbosely expressed value' : forma & formb
                            })
    return table



def delta_lexicon(df):
    """Returns all the shortest descriptions for an entire lexicon

    """

    table = []
    lexemes = set(df['lexeme'])
    for lexeme in lexemes:
        df2 = df[df['lexeme']== lexeme]
        formativelist = set(zip(df2.tier, df2.slot, df2.formative))
        for form in formativelist:
            description = shortest_description(form[0], form[1], form[2], lexeme, df2)
            table.append({
                'lexeme' : lexeme ,
                'tier' : form[0] ,
                'slot' : form[1] ,
                'formative' : form[2] ,
                'minimal description' : description ,
            })
    return table



def classify_unique(df):
    """Classifies all values in a lexicon (dataframe) with regards to
    uniqueness.

    """
    table = []
    lexemes = set(df['lexeme'])

    for lexeme in lexemes:
        dflex = df[df['lexeme']== lexeme]
        valuelist = dflex.cell.str.split(".").apply(set).tolist() #gets set of values for a given lexeme but
        valueset = set()
        for cell in valuelist:
            for value in cell:
                valueset.add(value)

        deltatable = delta_lexicon(dflex) #gets a list of the maximal delta for each formative

        for value in valueset: #goes through and makes a list of all for the formatives which have value in their minimum delta
            examplelist = []
            for delta in deltatable:
                for cell in delta['minimal description']:
                    if type(cell) != str:
                        if value in cell:
                            examplelist.append({'lexeme':lexeme,'value':value, 'slot' : delta['slot'],'form' : delta['formative']})
            examplelistunique = list({str(i):i for i in examplelist}.values()) #makes list unique
            if len(examplelistunique) == 1: #shows those which have just one formative...
                for x in examplelistunique:
                    table.append(
                        x
                    )

    return table



def classify_allomorphy(df):
    """Classifies all values in a lexicon (dataframe) with regards to
    uniqueness.

    Crucially, treats formatives which have the same distribution with
    regard to a value as a group (to exclude ME.)

    This excludes all ME but sould it treat all VE as a 'single allomorph'?
    """
    table = []
    lexemes = set(df['lexeme'])

    for lexeme in lexemes:
        dflex = df[df['lexeme']== lexeme]
        valuelist = dflex.cell.str.split(".").apply(frozenset).tolist() #gets set of values for a given lexeme but
        valueset = set()
        for cell in valuelist:
            for value in cell:
                valueset.add(value)

        deltatable = delta_lexicon(dflex) #gets a list of the maximal delta for each formative

        for value in valueset: #goes through and makes a list of all for the formatives which have value in their minimum delta
            examplelist = []
            for delta in deltatable:
                celllist = []
                for cell in delta['minimal description']:
                    if type(cell) != str:
                        if value in cell:
                            celllist.append(cell)
                if len(celllist) > 0:
                    examplelist.append({'lexeme':lexeme,'value':value, 'tier' : delta['tier'], 'slot' : delta['slot'],'form' : delta['formative'], 'cells' : celllist})
            #examplelistunique = list({str(i):i for i in examplelist}.values()) #makes list unique
            if len(examplelist) > 0:
                #THIS IS NON-UNIQUE EXPONENCE!! (use this for the others)
                #below checks to see if their distributions are identical:
                deltaset = set()
                for delta in examplelist:
                    for cell in delta['cells']:
                        deltaset.add(cell)

                #This one groups those with identical distribution with respect to a feature value:

                if len(deltaset) > 1:
                    celllist2 = []
                    examplelist.sort(key=itemgetter('cells'))

                    for key, value2 in itertools.groupby(examplelist, lambda item: item['cells']):
                        val2list = []
                        for x in value2:
                            form2 = (x['tier'], x['slot'], x['form'])
                            val2list.append(form2)
                        celllist2.append(val2list)

                    cellswithv = set(valuelist)
                    cellcount = sum(value in cell for cell in cellswithv)

                    table.append({
                        'lexeme' : lexeme,
                        'value' : value,
                        '# allomorphs' : len(celllist2),
                        '% of allomorphs to cells containing v' : len(celllist2)/cellcount *100 ,
                        'forms' : celllist2

                    })
    return table





def classify_VE(df):
    """Classifies all values in a lexicon (dataframe) with regards to
    verbose exponence.

    """
    table = []
    lexemes = set(df['lexeme'])

    for lexeme in lexemes:
        dflex = df[df['lexeme']== lexeme]
        valuelist = dflex.cell.str.split(".").apply(frozenset).tolist() #gets set of values for a given lexeme but
        valueset = set()
        for cell in valuelist:
            for value in cell:
                valueset.add(value)

        deltatable = delta_lexicon(dflex) #gets a list of the maximal delta for each formative

        for value in valueset: #goes through and makes a list of all for the formatives which have value in their minimum delta
            examplelist = []
            for delta in deltatable:
                celllist = []
                for cell in delta['minimal description']:
                    if type(cell) != str:
                        if value in cell:
                            celllist.append(cell)
                if len(celllist) > 0:
                    examplelist.append({'lexeme':lexeme,'value':value, 'tier' : delta['tier'], 'slot' : delta['slot'],'form' : delta['formative'], 'cells' : celllist})
            #examplelistunique = list({str(i):i for i in examplelist}.values()) #makes list unique
            if len(examplelist) > 0:
                print(examplelist)

def classify_VE(df):
    """Classifies all values in a lexicon (dataframe) with regards to
    verbose exponence.

    """
    table = []
    lexemes = set(df['lexeme'])

    for lexeme in lexemes:
        dflex = df[df['lexeme']== lexeme]
        valuelist = dflex.cell.str.split(".").apply(frozenset).tolist() #gets set of values for a given lexeme but
        #dflex['celllist'] = valuelist
        valueset = set()
        for cell in valuelist:
            for value in cell:
                valueset.add(value)

        deltatable = delta_lexicon(dflex) #gets a list of the maximal delta for each formative

        for value in valueset: #goes through and makes a list of all for the formatives which have value in their minimum delta
            examplelist = []
            for delta in deltatable:
                celllist = []
                for cell in delta['minimal description']:
                    if type(cell) != str:
                        if value in cell:
                            celllist.append(cell)
                if len(celllist) > 0:
                    examplelist.append({'lexeme':lexeme,'value':value, 'tier' : delta['tier'], 'slot' : delta['slot'],'form' : delta['formative'], 'cells' : celllist})

            if len(examplelist) > 0: #examplelist = list of formative for value
                locations = [idx for idx, cell in enumerate(valuelist) if cell > {value}] #locations of cells containing value
                wordlist = dflex.iloc[locations] #list of words for value
                #for each word... list all the formatives which co-occur

                for word in set(zip(wordlist.cell,wordlist.form)):
                    #print(wordlist)
                    dfword = wordlist[(wordlist['cell'] == word[0]) & (wordlist['form'] == word[1])]
                    formativelist = []
                    for formative in examplelist:
                        formtuple = (formative['tier'], formative['slot'], formative['form'])
                        if formtuple in zip(dfword.tier, dfword.slot, dfword.formative):
                            formativelist.append(formtuple)
                    if len(formativelist) > 1:
                        table.append({
                            'lexeme': lexeme,
                            'value': value,
                            'word' : word,
                            'formatives' : (formativelist),
                            '# formatives' : len(formativelist)
                            })
    return table
