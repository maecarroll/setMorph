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
    count_elts(exponents, "exponence", "# values")
    # simple exponence if = 1
    # syncretism if > 1
    max_dimensions = df["celllist"].fillna("").apply(len).max()
    classify_cumulation(exponents, max_dimensions)

    values_words = values_per_word(df, exponents)
    count_elts(values_words, "formative", "# formatives")

    values = values_table(values_words)
    count_elts(values, "formative", "# formatives")
    classify_allomorphy(df, values)

    ## Export
    values_words.formative = values_words.formative \
        .apply(lambda x: ' '.join([str(f) for f in x]))
    values_words.to_csv(output_prefix + "_values_per_word.csv", index=False)

    values.formative = values.formative \
        .apply(lambda x: ' '.join([str(f) for f in x]))
    values.formatives_by_word = values.formatives_by_word \
        .apply(
        lambda words: " ".join(f"#{' '.join([str(f) for f in w])}#" for w in words))
    values.to_csv(output_prefix + "_values.csv", index=False)

    exponents.exponence = exponents.exponence.apply(format_values)
    exponents.dist_a = exponents.dist_a.apply(format_values)
    exponents["cumulative cells"] = exponents["cumulative cells"].apply(format_values)
    exponents.to_csv(output_prefix + "_formatives.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("forms", type=str, help="Forms table")
    parser.add_argument("features", type=str, help="Features table")
    parser.add_argument("output", type=str, help="Output path and prefix")
    args = parser.parse_args()
    analyze_exponence(args.forms, args.features, args.output)
