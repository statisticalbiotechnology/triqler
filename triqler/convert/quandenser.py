'''
Create Triqler input files by combining Percolator output with the feature 
groups from Quandenser. Requires Triqler and NumPy to be installed.

If Percolator output is unavailable, one can mimick this format by providing
a tab-delimited file with the following columns (including a header row):

PSMId <tab> score <tab> q-value <tab> posterior_error_prob <tab> peptide <tab> proteinIds

The q-value and posterior_error_prob can be set to 0, as they are not used 
here. Furthermore, the file should be sorted by the score column, where higher 
scores indicate more confident hits and the highest score is on top of the list.
'''

from __future__ import print_function

import os
import sys
import numpy as np
import collections

from .. import parsers
from ..triqler import __version__, __copyright__

from . import normalize_intensities as normalize
from . import helpers

def main():
  print('Triqler.convert.quandenser version %s\n%s' % (__version__, __copyright__))
  print('Issued command:', os.path.basename(__file__) + " " + " ".join(map(str, sys.argv[1:])))
  
  args, params = parseArgs()
  
  convertQuandenserToTriqler(args.file_list_file, args.in_file, args.psm_files.split(","), args.out_file, params)

def parseArgs():
  import argparse
  apars = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  requiredNamed = apars.add_argument_group('required arguments')
  
  apars.add_argument('in_file', default=None, metavar = "IN_FILE",
                     help='''Quandenser output file with feature groups.
                          ''')
  
  requiredNamed.add_argument('--file_list_file', metavar='L', 
                     help='Simple text file with spectrum file names in first column and condition in second column.',
                     required = True)
  
  requiredNamed.add_argument('--psm_files', metavar='TARGET,DECOY', 
                     help='Percolator PSM output files, separated commas. Both target and decoy output files are needed, with the target file(s) specified first.',
                     required = True)
  
  apars.add_argument('--out_file', default = "triqler_input.tsv", metavar='OUT', 
                     help='''Path to triqler input file (writing in TSV format).
                          ''')
  
  apars.add_argument('--skip_normalization',
                     help='Skip retention-time based intensity normalization.',
                     action='store_true')
  
  apars.add_argument('--retain_unidentified',
                     help='Keeps features without identification in the Triqler input file.',
                     action='store_true')
                     
  apars.add_argument('--skip_link_pep',
                     help='Skips the linkPEP column from match-between-runs output.',
                     action='store_true')
  
  # ------------------------------------------------
  args = apars.parse_args()
  
  params = dict()
  params['simpleOutputFormat'] = args.skip_link_pep
  params['skipNormalization'] = args.skip_normalization
  params['retainUnidentified'] = args.retain_unidentified
  params['plotScatter'] = False
  
  return args, params
  
def convertQuandenserToTriqler(fileListFile, clusterQuantFile, psmsOutputFiles, peptQuantRowFile, params):
  fileInfoList = parsers.parseFileList(fileListFile)
  
  if not params['skipNormalization']:
    clusterQuantFileNormalized = clusterQuantFile.replace(".tsv", ".normalized.tsv")
    if not os.path.isfile(clusterQuantFileNormalized):
      print("Applying retention-time dependent intensity normalization")
      minRunsObservedIn = len(fileInfoList) / 3 + 1
      normalize.normalizeIntensitiesRtimeBased(clusterQuantFile, clusterQuantFileNormalized, minRunsObservedIn, plotScatter = params['plotScatter'])
    else:
      print("Reusing previously generated normalized feature group file:", clusterQuantFileNormalized, ". Remove this file to redo normalization")
    clusterQuantFile = clusterQuantFileNormalized
  else:
    print("Skipping retention-time dependent intensity normalization")
  
  specToPeptideMap = helpers.parsePsmsPoutFiles(psmsOutputFiles)
  
  printTriqlerInputFile(fileInfoList, clusterQuantFile, peptQuantRowFile, specToPeptideMap, params)
  
def parsePeptideLinkPEP(peptLinkPEP):
  spectrumIdx, linkPEP = peptLinkPEP.split(";")
  return int(spectrumIdx), float(linkPEP)

def printTriqlerInputFile(fileInfoList, clusterQuantFile, quantRowFile, specToPeptideMap, params):
  print("Parsing cluster quant file")
  
  writer = parsers.getTsvWriter(quantRowFile)
  if params['simpleOutputFormat']:
    writer.writerow(parsers.TriqlerSimpleInputRowHeaders)
  else:
    writer.writerow(parsers.TriqlerInputRowHeaders)
  
  featureClusterRows = list()
  spectrumToFeatureMatch = dict() # stores the best peptideQuantRow per (peptide, spectrumIdx)-pair
  for featureClusterIdx, featureCluster in enumerate(parsers.parseFeatureClustersFile(clusterQuantFile)):
    if featureClusterIdx % 50000 == 0:
      print("Processing feature group", featureClusterIdx + 1)
    
    rows = list()
    for pc in featureCluster:
      fileIdx = int(pc.fileName)
      for peptLinkPEP in pc.peptLinkPEPs.split(","):
        spectrumIdx, linkPEP = parsePeptideLinkPEP(peptLinkPEP)
        peptide, proteins, searchScore, charge = specToPeptideMap(spectrumIdx)
        if pc.intensity > 0.0 and linkPEP < 1.0 and (params["retainUnidentified"] or peptide != "NA"):
          # run condition charge spectrumId linkPEP featureClusterId search_score intensity peptide proteins
          run, condition, sample, fraction = fileInfoList[fileIdx]
          row = parsers.TriqlerInputRow(sample, condition, charge, spectrumIdx, linkPEP, featureClusterIdx, searchScore, pc.intensity, peptide, proteins)
          rows.append(row)
    
    newRows = list()
    rows = sorted(rows, key = lambda x : (x.run, x.spectrumId, x.linkPEP, -1*x.searchScore, -1*x.intensity))
    prevKey = (-1, -1)
    bestSearchScore = -1e9
    for row in rows:
      if prevKey == (row.run, row.spectrumId):
        if row.searchScore > bestSearchScore:
          bestSearchScore = row.searchScore
          newRows.append(row)
      else:
        newRows.append(row)
        prevKey = (row.run, row.spectrumId)
    
    for row in newRows:
      if params['simpleOutputFormat']:
        writer.writerow(row.toSimpleList())
      else:
        writer.writerow(row.toList())

if __name__ == "__main__":
   main()
