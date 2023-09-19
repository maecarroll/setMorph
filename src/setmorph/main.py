#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .setmorph import read_paradigms, read_features, \
    get_reals, get_exponents, get_real_per_word, \
    classify_cumulation, classify_allomorphy
import argparse


def format_cell(vals):
    return ".".join(vals)

def format_feature_values(vals):
    return " ".join(format_cell(v) for v in vals)

def format_formatives(formatives):
    return " ".join(str(f) for f in sorted(formatives))

def format_allo_sets(f_per_w):
    return " ".join("-".join(str(f) for f in sorted(w)) for w in f_per_w)


def analyse_exponence(forms_path, features_path, output_prefix):
    df = read_paradigms(forms_path)
    fs = read_features(features_path)
    print("Computing form<->meaning mapping...")
    exponents = get_exponents(df, fs) # rows are formatives, gives dist, exp, vals
    print("Finding all realizations of a value...")
    reals = get_reals(exponents) # rows are value, lexme pairs, gives real
    print("Finding all realizations of a value, per word...")
    real_w = get_real_per_word(df, reals) # rows are word/value pairs, give real_w

    print("Classifying non-canonical phenomena...")
    # Add sets related to cumulation
    classify_cumulation(exponents)

    # Measuring allomorphy requires a set derived from
    # real, but where elements are sets of realization,
    # for each word
    reals = classify_allomorphy(reals, real_w)

    ## Real table formatting
    reals["real"] = reals["real"].apply(format_formatives)
    reals["allomorphic_sets"] = reals["allomorphic_sets"].apply(format_allo_sets)
    reals.reset_index(drop=False, inplace=True)

    ## Real_w table formatting
    real_w["cell"] = real_w["cell"].apply(format_cell)
    for col in ["wordform", "real_w"]:
        real_w[col] = real_w[col].apply(format_formatives)

    ## Exponence table formatting
    for col in ["exponence", "dist", "cumulative",
                "cumulative_cells"]:
        exponents[col] = exponents[col].apply(format_feature_values)
    for col in ["vals"]:
        exponents[col] = exponents[col].apply(format_formatives)

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
    analyse_exponence(args.forms, args.features, args.output)
