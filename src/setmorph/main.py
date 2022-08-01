#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .setmorph import read_paradigms, read_features
from .classify import *
import argparse


def format_values(vals):
    return " ".join(".".join(v) for v in vals)


def analyze_exponence(forms_path, features_path, output_prefix):
    df = read_paradigms(forms_path)
    fs = read_features(features_path)

    exponents = find_exponents(df, fs)
    classify_simple(exponents)
    classify_syn(exponents)
    max_dimensions = df["celllist"].fillna("").apply(len).max()
    classify_cumulation(exponents, max_dimensions)

    values_words = values_per_word(df, exponents)
    count_formatives(values_words)

    values = values_table(values_words)
    count_formatives(values)
    classify_allomorphy(df, values)

    ## Export
    values_words.formatives = values_words.formatives \
        .apply(lambda x: ' '.join([str(f) for f in x]))
    values_words.to_csv(output_prefix + "_values_per_word.csv")

    values.formatives = values.formatives \
        .apply(lambda x: ' '.join([str(f) for f in x]))
    values.formatives_by_word = values.formatives_by_word \
        .apply(
        lambda words: " ".join(f"#{' '.join([str(f) for f in w])}#" for w in words))
    values.to_csv(output_prefix + "_values.csv")

    exponents.exponence = exponents.exponence.apply(format_values)
    exponents.dist_a = exponents.dist_a.apply(format_values)
    exponents["cumulative cells"] = exponents["cumulative cells"].apply(format_values)
    exponents.to_csv(output_prefix + "_formatives.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("forms", type=str, help="Forms table")
    parser.add_argument("features", type=str, help="Features table")
    parser.add_argument("output", type=str, help="Output path and prefix")
    args = parser.parse_args()
    analyze_exponence(args.forms, args.features, args.output)
