import itertools

def feat_set (sig, feat_df):
    feat_set = set()
    for value in feat_df[feat_df['feature']==sig]['value']:
        feat_set.add(value)
    return feat_set


def parse_cell (df):
    cells = df['cell']
    cellSetList = []
    for cell in cells:
        cell = cell.split('.')
        cell = frozenset(cell)
        cellSetList.append(cell)
    return cellSetList


def parse_formatives (df):

    formativelist = list(zip(df.tier,df.slot,df.formative))
    df['formtuples'] = formativelist
    wordformset = []
    for wordform in df['form']:
        wordform_asset = frozenset(df['formtuples'][df['form'] == wordform])
        wordformset.append(wordform_asset)
    return wordformset

def parse_formatives_lex (df):

    formativelist = list(zip(df.lexeme,df.tier,df.slot,df.formative))
    df['formtuples'] = formativelist
    wordformset = []
    for wordform in df['form']:
        wordform_asset = frozenset(df['formtuples'][df['form'] == wordform])
        wordformset.append(wordform_asset)
    return wordformset


def dist (tier, pos, form, lex, df):
    dist = df[(df['lexeme'] == lex) &
              (df['tier'] == tier) &
              (df['slot'] == pos) &
              (df['formative'] == form)]
    return dist


def real (val, lex, df): ## Assumes that all feature values are unique
    cells = parse_cell(df)
    df['celllist'] = cells
    val = {val}
    real = df[(df['lexeme'] == lex) &
              (df['celllist'] & val)]
    return real


def shortest_description (tier, pos, form, lex, df):
    cells = parse_cell(df)
    df['celllist'] = cells
    description = set()
    dista = dist(tier, pos, form, lex, df)

    bigdelta = []

    for cell in cells:
        bigdelta.append(
            {frozenset(delta) for i in range (1, len(cell)) for delta in itertools.combinations (cell, i)}
        )

    deltaset = set()
    for combo in bigdelta:
        for delta in combo:
            deltaset.add(delta)


    for delta in deltaset:
        cellsdelta = dista[dista['celllist'] & delta]# DF with all the cells with delta and a
        totalcellsdelta = df[df['celllist'] & delta] # DF with ALL the cells with delta

        cellsdeltalist = set(cellsdelta['celllist'])
        totalcellsdeltalist = set(totalcellsdelta['celllist'])


        if len(cellsdelta['celllist']) != 0:
            if cellsdeltalist == totalcellsdeltalist:
                description.add(delta)
            else:
                description.update(cellsdeltalist)


    combo = itertools.combinations(description, 2)
    for a, b in combo:
        if a < b:
            description.remove(b)
        elif b < a:
            description.remove(a)

    #given some form - what are all the cells that match that description
    return description


# def shortest_form (val, lex, df):
#     df = df[df['lexeme'] == lex].copy()
#     wordforms = parse_formatives(df)
#
#     # 1. get all the formatives
#     # 2. organise them by wordform
#     # 3. wordforms as sets of formatives == cells
#     # 4. Get bigalpha from bigdelta with cells -> formatives
#     # 5. for each alpha in big alpha - if the words which are described by are all the words in the word, then good 0! its an exponent of that feature.
#     # 6. reduce redundnacy from big alpha
#
#     df['wordform'] = wordforms
#     reala = real(val, lex, df)
#
#
#     bigalpha = []
#
#     wordformseta = reala['wordform']
#
#     for wordform in wordformseta:
#         bigalpha.append(
#             alpha for i in range (1, len(wordform)) for alpha in itertools.combinations (wordform, i)
#         )
#     alphaset = set()
#     for combo in bigalpha:
#         for alpha in combo:
#             alphaset.add(alpha)
#
#     longest_description = set()
#
#
#     for alpha in alphaset:
#         alpha = frozenset(alpha)
#         wordformsalpha = reala[reala['wordform'] > alpha]
#         totalwordformsalpha = df[df['wordform'] >= alpha]        #
#
#         wordformsalphalist = set(wordformsalpha['wordform'])
#         totalwordformsalphalist = set(totalwordformsalpha['wordform'])
#
#         if len(wordformsalpha['wordform']) != 0:
#             if wordformsalphalist == totalwordformsalphalist:
#                 longest_description.add(alpha)
#             else:
#                 longest_description.update(wordformsalphalist)
#
#
#
#     all_combinations = []
#     for r in range(len(description) + 1):
#         combinations_object = itertools.combinations(description, r)
#         combinations_list = list(combinations_object)
#         all_combinations += combinations_list
#
#     #print(all_combinations)
#     desc2 = set()
#
#     for alpha in all_combinations:
#         #print(alpha)
#         alpha = frozenset(alpha)
#
#         wordformsalpha = reala[reala['wordform'] > alpha]
#         totalwordformsalpha = df[df['wordform'] >= alpha]        #
#
#         wordformsalphalist = set(wordformsalpha['wordform'])
#         totalwordformsalphalist = set(totalwordformsalpha['wordform'])
#
#         if len(wordformsalpha['wordform']) != 0:
#             if wordformsalphalist == totalwordformsalphalist:
#                 desc2.add(alpha)
#             else:
#                 desc2.update(wordformsalphalist)
#
#         ## QUESTION - How is this different to the older version in terms of the list? This was supposed to be a list of all possible combinations of combinations... so that a
#         ## Obviously the minlist needs to come earlier in the assessment
#
#
#     minlist = [min(desc2, key=len)]
#
#     # for a, b in combo:
#     #         if a < b:
#     #             if b in description:
#     #                 description.remove(b)
#     #         elif b < a:
#     #             if a in description:
#     #                 description.remove(a)
#
#     print(minlist)
#     # given some value what are all the wordforms that match that description
#     return description
