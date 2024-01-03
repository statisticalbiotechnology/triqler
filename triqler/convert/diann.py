"""
Create Triqler input files from DIA-NN output files.
"""

import os
import sys
import glob
from typing import Dict

import numpy as np
import pandas as pd

from ..triqler import __version__, __copyright__
from .. import parsers


def main():
    print("triqler.convert.diann version %s\n%s" % (__version__, __copyright__))
    print(
        "Issued command:",
        os.path.basename(__file__) + " " + " ".join(map(str, sys.argv[1:])),
    )

    args, params = parseArgs()

    # hack for windows
    if len(args.in_file) == 1 and "*" in args.in_file[0]:
        args.in_file = glob.glob(args.in_file[0])

    diann_to_triqler(args.in_file, args.file_list_file, args.out_file, params)


def parseArgs():
    import argparse

    apars = argparse.ArgumentParser(
        description="Converts DIA-NN output files to Triqler input format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    requiredNamed = apars.add_argument_group("required arguments")

    apars.add_argument(
        "in_file",
        default=None,
        metavar="IN_FILE",
        help="""DIA-NN output file
                          """,
    )

    requiredNamed.add_argument(
        "--file_list_file",
        metavar="L",
        help="""Simple tab separated file with run names in first column and condition in second column. 
                             The run names should be identical to the entries in the "Run" column of the DIA-NN output file.
                             """,
        required=True,
    )

    apars.add_argument(
        "--out_file",
        default="triqler_input.tsv",
        metavar="OUT",
        help="""Path to triqler input file (writing in TSV format).
                          """,
    )

    # ------------------------------------------------
    args = apars.parse_args()

    params = dict()
    return args, params


def diann_to_triqler(
    diann_file_path: str, file_list_file: str, triqler_input_file: str, params: Dict
):
    file_list_df = parse_file_list(file_list_file)

    sample_mapper = dict(zip(file_list_df["run"], file_list_df["sample"]))
    condition_mapper = dict(zip(file_list_df["run"], file_list_df["condition"]))

    df = pd.read_csv(diann_file_path, sep="\t")

    df["run"] = df["Run"].map(sample_mapper)
    df["condition"] = df["Run"].map(condition_mapper)
    df["charge"] = df["Precursor.Charge"]
    df["searchScore"] = -np.log(df["Q.Value"])
    df["intensity"] = df["Precursor.Quantity"]
    df["peptide"] = df["Stripped.Sequence"]
    df["proteins"] = df["Protein.Ids"]
    triqler_input_df = df[
        [
            "run",
            "condition",
            "charge",
            "searchScore",
            "intensity",
            "peptide",
            "proteins",
        ]
    ]

    triqler_input_df.to_csv(triqler_input_file, sep="\t", index=False)


def parse_file_list(file_list_file: str):
    file_list_df = pd.read_csv(file_list_file, sep="\t", header=None)

    if len(file_list_df.columns) < 2:
        raise ValueError(
            "Too few columns present in file list mapping, need at least 2 columns: run, condition"
        )

    if len(file_list_df.columns) > 4:
        raise ValueError(
            "Too many columns present in file list mapping, can at most have 4 columns: run, condition, sample, fraction"
        )

    file_list_df.columns = ["run", "condition", "sample", "fraction"][
        : len(file_list_df.columns)
    ]

    if "sample" not in file_list_df.columns:
        file_list_df["sample"] = file_list_df["run"]

    if "fraction" not in file_list_df.columns:
        file_list_df["fraction"] = -1

    return file_list_df


if __name__ == "__main__":
    main()
