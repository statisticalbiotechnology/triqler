'''
Create Triqler input files from DIA-NN output files.
'''

import os
import sys
import glob
from typing import Dict

import numpy as np
import pandas as pd

from ..triqler import __version__, __copyright__


def main():
  print('triqler.convert.diann version %s\n%s' % (__version__, __copyright__))
  print('Issued command:', os.path.basename(__file__) + " " + " ".join(map(str, sys.argv[1:])))
  
  args, params = parseArgs()
  
  # hack for windows
  if len(args.in_file) == 1 and '*' in args.in_file[0]:
    args.in_file = glob.glob(args.in_file[0])
  
  diann_to_triqler(args.in_file, args.out_file, params)


def parseArgs():
  import argparse
  apars = argparse.ArgumentParser(
      description='Converts DIA-NN output files to Triqler input format.',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  apars.add_argument('in_file', default=None, metavar = "IN_FILE",
                     help='''DIA-NN output file
                          ''')
  
  apars.add_argument('--out_file', default = "triqler_input.tsv", metavar='OUT', 
                     help='''Path to triqler input file (writing in TSV format).
                          ''')
  
  # ------------------------------------------------
  args = apars.parse_args()
  
  params = dict()
  return args, params


def diann_to_triqler(diann_file_path: str, triqler_input_file: str, params: Dict):
  df = pd.read_csv(diann_file_path, sep='\t')
  
  run_mapper = lambda x : x.split("_")[5]
  condition_mapper = lambda x : x.split("_")[8]
  
  df["run"] = df["Run"].map(run_mapper)
  df["condition"] = df["Run"].map(condition_mapper)
  df["charge"] = df["Precursor.Charge"]
  df["searchScore"] = -np.log(df["Q.Value"])
  df["intensity"] = df['Precursor.Quantity']
  df["peptide"] = df["Stripped.Sequence"]
  df["proteins"] = df["Protein.Ids"]
  triqler_input_df = df[["run", "condition", "charge", "searchScore", "intensity", "peptide", "proteins"]]

  triqler_input_df.to_csv(triqler_input_file, sep='\t', index=False)


if __name__ == "__main__":
   main()
