import itertools


def feat_set(sig, feat_df):
    feat_set = set()
    for value in feat_df[feat_df['feature'] == sig]['value']:
        feat_set.add(value)
    return feat_set


def parse_cell(df):
    cells = df['cell']
    cellSetList = []
    for cell in cells:
        cell = cell.split('.')
        cell = frozenset(cell)
        cellSetList.append(cell)
    return cellSetList


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
    cells = parse_cell(df)
    df['celllist'] = cells
    description = set()
    dista = dist(tier, pos, form, lex, df)

    bigdelta = []

    for cell in cells:
        bigdelta.append(
            {frozenset(delta) for i in range(1, len(cell)) for delta in
             itertools.combinations(cell, i)}
        )

    deltaset = set()
    for combo in bigdelta:
        for delta in combo:
            deltaset.add(delta)

    for delta in deltaset:
        cellsdelta = dista[
            dista['celllist'] & delta]  # DF with all the cells with delta and a
        totalcellsdelta = df[df['celllist'] & delta]  # DF with ALL the cells with delta

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

    # given some form - what are all the cells that match that description
    return description
