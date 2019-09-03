#!/usr/bin/python

from __future__ import print_function

import os
import sys
import warnings

from ..triqler import __version__, __copyright__
from .. import hyperparameters
from .. import parsers

def main():
  print('Triqler.distribution.plot_hyperparameter_fits version %s\n%s' % (__version__, __copyright__))
  print('Issued command:', os.path.basename(__file__) + " " + " ".join(map(str, sys.argv[1:])))
  
  args, params = parseArgs()
  
  params['warningFilter'] = "ignore"
  with warnings.catch_warnings():
    warnings.simplefilter(params['warningFilter'])
    plotHyperparameterFits(args.in_file, params)

def parseArgs():
  import argparse
  apars = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  apars.add_argument('in_file', default=None, metavar = "IN_FILE",
                     help='''Triqler input file.
                          ''')
  
  apars.add_argument('--decoy_pattern', default = "decoy_", metavar='P', 
                     help='Prefix for decoy proteins.')
  
  apars.add_argument('--no_plots',
                     help='Only print out hyperparameter estimates, without plotting the empirical and fitted distributions.',
                     action='store_true')
  
  # ------------------------------------------------
  args = apars.parse_args()
  
  params = dict()
  params['decoyPattern'] = args.decoy_pattern
  params['skipPlots'] = args.no_plots
    
  return args, params
  
def plotHyperparameterFits(triqlerInputFile, params):  
  if not os.path.isfile(triqlerInputFile):
    sys.exit("Could not locate input file %s. Check if the path to the input file is correct." % triqlerInputFile)
  
  peptQuantRowFile = triqlerInputFile + ".pqr.tsv"
  if not os.path.isfile(peptQuantRowFile):
    sys.exit("Could not locate peptide quantification file %s. Run triqler to generate this file." % peptQuantRowFile)
  
  _, params['groups'], _, peptQuantRows = parsers.parsePeptideQuantFile(peptQuantRowFile)
  
  if not params['skipPlots']:
    import matplotlib
    matplotlib.rcParams['axes.unicode_minus'] = False
  
  print("")
  print("N.B. naming of hyperparameters differs from notation in the paper:")
  print("  muDetect = mu_m; sigmaDetect = sigma_m;")
  print("  muXIC = mu_f; sigmaXIC = sigma_f;")
  print("  muProtein = mu_y; sigmaProtein = sigma_y;")
  print("  muFeatureDiff = mu_d; sigmaFeatureDiff = sigma_d;")
  print("  shapeInGroupStdevs = alpha_s; scaleInGroupStdevs = beta_s")
  print("")
  
  hyperparameters.fitPriors(peptQuantRows, params, plot = not params['skipPlots'])
  if not params['skipPlots']:
    import matplotlib.pyplot as plt
    plt.show()
  
if __name__ == "__main__":
  main()
