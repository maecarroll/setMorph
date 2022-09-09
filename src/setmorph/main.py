#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .setmorph import read_paradigms, read_features, \
    get_reals, get_exponents, get_real_per_word, get_words_table
from .classify import classify_cumulation, classify_allomorphy, count_elts
import argparse


def format_feature_values(vals):
    return " ".join(".".join(v) for v in vals)

def format_formatives(formatives):
    return " ".join(str(f) for f in sorted(formatives))

def analyze_exponence(forms_path, features_path, output_prefix):
    df = read_paradigms(forms_path)
    fs = read_features(features_path)

    exponents = get_exponents(df, fs) # rows are formatives, gives dist, exp, vals
    reals = get_reals(exponents) # rows are value, lexme pairs, gives real
    words = get_words_table(df) # rows are words, gives wordform as set of formatives
    real_w = get_real_per_word(words, reals) # rows are word/value pairs, give real_w

    # if simple, |vals| == 1
    count_elts(exponents, "vals", "|vals|")

    # if syncretic, |exp| > 1
    count_elts(exponents, "exponence", "|exp|")

    # Add sets related to cumulation
    classify_cumulation(exponents)

    # unique exponence if |real| == 1
    count_elts(reals, "real", "|real|")

    # verbose exponence if |real_w| > 1
    count_elts(real_w, "real_w", "|real_w|")

    # Measuring allomorphy requires a set derived from
    # real, but where elements are sets of realization,
    # for each word
    reals = classify_allomorphy(reals, real_w)

    ## Real table formatting
    for col in ["real", "allomorphic_sets"]:
        reals[col] = reals[col].apply(format_formatives)

    ## Real_w table formatting
    real_w["cell"] = real_w["cell"].apply(format_feature_values)
    for col in ["wordform", "real_w"]:
        real_w[col] = real_w[col].apply(format_formatives)

    ## Exponence table formatting
    for col in ["exponence", "dist", "vals", "cumulative_vals",
                "cumulative_cells"]:
        exponents[col] = exponents[col].apply(format_feature_values)

    ## Export
    real_w.to_csv(output_prefix + "_values_per_word.csv", index=False)
    reals.to_csv(output_prefix + "_values.csv", index=False)
    exponents.to_csv(output_prefix + "_formatives.csv", index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("forms", type=str, help="Forms table")
    parser.add_argument("features", type=str, help="Features table")
    parser.add_argument("output", type=str, help="Output path and prefix")
    args = parser.parse_args()
    analyze_exponence(args.forms, args.features, args.output)
