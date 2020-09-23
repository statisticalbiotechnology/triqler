'''
Create Triqler input files by combining Percolator output with the feature 
groups from Dinosaur. Requires Triqler and NumPy to be installed.

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
import glob
import numpy as np
import collections

from .. import parsers
from ..triqler import __version__, __copyright__

from . import helpers
from . import percolator

def main():
  print('Triqler.convert.dinosaur version %s\n%s' % (__version__, __copyright__))
  print('Issued command:', os.path.basename(__file__) + " " + " ".join(map(str, sys.argv[1:])))
  
  args, params = parseArgs()
  
  # hack for windows
  if len(args.in_file) == 1 and '*' in args.in_file[0]:
    args.in_file = glob.glob(args.in_file[0])
  
  convertDinosaurToTriqler(args.file_list_file, args.in_file, args.psm_files.split(","), args.out_file, params)

def parseArgs():
  import argparse
  apars = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  requiredNamed = apars.add_argument_group('required arguments')
  
  apars.add_argument('in_file', default=None, metavar = "IN_FILE", nargs='*',
                     help='''Files containing the mapping of scan numbers to precursor information from the add-dinosaur-precursors script. To easily specify multiple files one can use wildcards, e.g. dinosaur_mapping_files/*.txt
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
  
  # ------------------------------------------------
  args = apars.parse_args()
  
  params = dict()
  params['simpleOutputFormat'] = True
  params['skipNormalization'] = args.skip_normalization
  params['plotScatter'] = False
  params['skipMBR'] = True
  
  return args, params
  
def convertDinosaurToTriqler(fileListFile, mappedPrecursorFiles, psmsOutputFiles, triqlerInputFile, params):
  fileInfoList = parsers.parseFileList(fileListFile)
  
  peptideToFeatureMap = parseDinosaurMapFiles(mappedPrecursorFiles, fileInfoList, psmsOutputFiles)
  
  rTimeArrays, factorArrays = helpers.getNormalizationFactorArrays(peptideToFeatureMap, fileInfoList, params)
  
  helpers.writeTriqlerInputFile(triqlerInputFile, peptideToFeatureMap, rTimeArrays, factorArrays, params)
  
  print("Finished writing Triqler input file:", triqlerInputFile)

def parseDinosaurMapFiles(mappedPrecursorFiles, fileInfoList, psmsOutputFiles):
  fileList, _, _, _ = zip(*fileInfoList)
  
  scanToFeatureMaps = collections.defaultdict(lambda : collections.defaultdict(list))
  for mappedPrecursorFile in mappedPrecursorFiles:
    # spMap: fileName scanNr precMz charge rTime intensity
    for spMap in parsers.parseMappedPrecursorFile(mappedPrecursorFile):
      baseFN = os.path.splitext(spMap.fileName)[0]
      scanToFeatureMaps[baseFN][spMap.scanNr].append(spMap)
  
  fileKey = -1
  linkPEP = 0.0
  peptChargeToFeatureMap = collections.defaultdict(list)
  for psmsOutputFile in psmsOutputFiles:
    print("Processing", psmsOutputFile)
    assignedPSMs, unassignedPSMs = 0, 0
    # psm: id filename scannr charge svm_score qvalue PEP peptide proteins
    for psm in percolator.parsePsmsPout(psmsOutputFile):
      if fileKey == -1:
        if isinstance(psm.filename, int):
          fileKey = 2 # crux percolator
        else:
          fileKey = 1 # standalone percolator
      
      if fileKey == 2:
        fileIdx = int(psm.filename)
      else:
        fileIdx = fileList.index(psm.filename)
      
      fileName = fileList[fileIdx]
      
      run, condition, sample, fraction = fileInfoList[fileIdx]
      calcMass = helpers.calcMass(psm.peptide)
      
      bestSpMap = (0.1, None) # do not map features with more than 0.1Da mass difference
      for spMap in scanToFeatureMaps[fileName][psm.scannr]:
        if spMap.charge != psm.charge:
          continue
        
        massDiff = abs(helpers.precMassFromPrecMz(spMap.precMz, spMap.charge) - calcMass)
        if massDiff < bestSpMap[0]:
          bestSpMap = (massDiff, spMap)
      
      if bestSpMap[1]:
        spMap = bestSpMap[1]
        peptChargePair = (psm.peptide, psm.charge)
        if peptChargePair in peptChargeToFeatureMap:
          featureClusterIdx = peptChargeToFeatureMap[peptChargePair][0][0].featureClusterId
        else:
          featureClusterIdx = len(peptChargeToFeatureMap)
        triqlerRow = parsers.TriqlerInputRow(sample, condition, psm.charge, psm.scannr, linkPEP, featureClusterIdx, psm.svm_score, spMap.intensity, psm.peptide, psm.proteins)
        peptChargeToFeatureMap[peptChargePair].append((triqlerRow, spMap.rTime, fraction))
        assignedPSMs += 1
      else:
        unassignedPSMs += 1
        #calcMz = helpers.precMzFromPrecMass(calcMass, psm.charge)
        #print("Assigned PSMs:", assignedPSMs, "Unassigned PSMs:", unassignedPSMs)
        #print(psm, calcMz)
        #print(scanToFeatureMaps[fileName][psm.scannr])
        #print("")
        #if unassignedPSMs > 100:
        #  sys.exit()
    print("Assigned PSMs:", assignedPSMs, "Unassigned PSMs:", unassignedPSMs)
  
  if len(peptChargeToFeatureMap) == 0:
    sys.exit("Error: Could not map any PSM to the appropriate dinosaur feature.")
  
  return peptChargeToFeatureMap

if __name__ == "__main__":
   main()
