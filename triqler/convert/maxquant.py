'''
Create Triqler input files by converting the evidence.txt output file from
MaxQuant.
'''

from __future__ import print_function

import sys
import os
import collections

import numpy as np

from .. import parsers
from ..triqler import __version__, __copyright__

from . import normalize_intensities as normalize

def main():
  print('Triqler.convert.maxquant version %s\n%s' % (__version__, __copyright__))
  print('Issued command:', os.path.basename(__file__) + " " + " ".join(map(str, sys.argv[1:])))
  
  args, params = parseArgs()
  
  convertMqToTriqler(args.file_list_file, args.in_file, args.out_file, params)

def parseArgs():
  import argparse
  apars = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  requiredNamed = apars.add_argument_group('required arguments')
  
  apars.add_argument('in_file', default=None, metavar = "IN_FILE",
                     help='''MaxQuant evidence.txt file.
                          ''')
  
  requiredNamed.add_argument('--file_list_file', metavar='L', 
                     help='Simple text file with spectrum file names in first column and condition in second column.',
                     required = True)
  
  apars.add_argument('--out_file', default = "triqler_input.tsv", metavar='OUT', 
                     help='''Path to triqler input file (writing in TSV format).
                          ''')
  
  apars.add_argument('--skip_normalization',
                     help='Skip retention-time based intensity normalization.',
                     action='store_true')
  
  #apars.add_argument('--skip_link_pep',
  #                   help='Skips the linkPEP column from match-between-runs output.',
  #                   action='store_true')
  
  # ------------------------------------------------
  args = apars.parse_args()
  
  params = dict()
  params['simpleOutputFormat'] = True
  params['skipNormalization'] = args.skip_normalization
  params['plotScatter'] = False
  
  return args, params
  
def convertMqToTriqler(fileListFile, mqEvidenceFile, triqlerInputFile, params):
  fileInfoList = parsers.parseFileList(fileListFile)
  fileList, _, sampleList, fractionList = zip(*fileInfoList)
  
  writer = parsers.getTsvWriter(triqlerInputFile)
  if params['simpleOutputFormat']:
    writer.writerow(parsers.TriqlerSimpleInputRowHeaders)
  else:
    writer.writerow(parsers.TriqlerInputRowHeaders)
    
  reader = parsers.getTsvReader(mqEvidenceFile)
  headers = next(reader) # save the header
  
  peptCol = headers.index('Modified sequence')
  idCol = headers.index('MS/MS scan number')
  fileCol = headers.index('Raw file')
  chargeCol = headers.index('Charge')
  intensityCol = headers.index('Intensity')
  proteinCol = headers.index('Leading proteins')
  scoreCol = headers.index('Score')
  postErrCol = headers.index('PEP')
  rtCol = headers.index('Retention time')
  
  fractionCol = headers.index('Fraction') if 'Fraction' in headers else -1
  experimentCol = headers.index('Experiment')
  
  print("Parsing MaxQuant evidence.txt file")
  peptideToFeatureMap = collections.defaultdict(list)
  for lineIdx, row in enumerate(reader):
    if lineIdx % 500000 == 0:
      print("  Reading line", lineIdx)
    
    proteins = row[proteinCol].split(";")
    
    linkPEP = 0.0
    key = (row[peptCol], row[chargeCol])
    
    fileIdx = fileList.index(row[fileCol])
    run, condition, sample, fraction = fileInfoList[fileIdx]
    if fraction == -1 and fractionCol != -1:
      sample, fraction = row[experimentCol], row[fractionCol]
    
    if key in peptideToFeatureMap:
      featureClusterIdx = peptideToFeatureMap[key][0][0].featureClusterId
    else:
      featureClusterIdx = len(peptideToFeatureMap)
    
    if row[intensityCol] == "" or float(row[scoreCol]) <= 0:
      continue
    
    triqlerRow = parsers.TriqlerInputRow(sample, condition, row[chargeCol], lineIdx, linkPEP, featureClusterIdx, np.log(float(row[scoreCol])), float(row[intensityCol]), row[peptCol], proteins)
    peptideToFeatureMap[key].append((triqlerRow, float(row[rtCol]), fraction))
  
  if not params['skipNormalization']:
    print("Applying retention-time dependent intensity normalization")
    minRunsObservedIn = len(set(sampleList)) / 3 + 1
    rTimeArrays, factorArrays = dict(), dict()
    for fraction in sorted(list(set(fractionList))):
      factorPairs = normalize.getIntensityFactorPairs(peptideToFeatureMap.values(), sortKey = lambda x : (x[2], x[0].run, -1*x[0].intensity, x[1]), minRunsObservedIn = minRunsObservedIn, fraction = fraction)
      print("Fraction:", fraction, "#runs:", len(factorPairs))
      if params['plotScatter']:
        normalize.plotFactorScatter(factorPairs)
      
      rTimeFactorArrays = normalize.getFactorArrays(factorPairs)
      
      rTimeArrays[fraction], factorArrays[fraction] = dict(), dict()
      for key in rTimeFactorArrays:
        rTimeArrays[fraction][key], factorArrays[fraction][key] = zip(*rTimeFactorArrays[key])
  else:
    print("Skipping retention-time dependent intensity normalization")
  
  for featureClusterIdx, featureCluster in enumerate(peptideToFeatureMap.values()):
    if featureClusterIdx % 50000 == 0:
      print("Processing feature group", featureClusterIdx + 1)
    newRows = selectBestScorePerRun(featureCluster)
    
    for (row, rTime, fraction) in newRows:
      if not params['skipNormalization']:
        newIntensity = normalize.getNormalizedIntensity(rTimeArrays[fraction][row.run], factorArrays[fraction][row.run], rTime, row.intensity)
        row = row._replace(intensity = newIntensity)
      
      if params['simpleOutputFormat']:
        writer.writerow(row.toSimpleList())
      else:
        writer.writerow(row.toList())

def selectBestScorePerRun(rows):
  newRows = list()
  rows = sorted(rows, key = lambda x : (x[0].run, x[0].spectrumId, x[0].linkPEP, -1*x[0].searchScore))
  prevKey = (-1, -1)
  bestSearchScore = -1e9
  for row in rows:
    if prevKey == (row[0].run, row[0].spectrumId):
      if row[0].searchScore > bestSearchScore:
        bestSearchScore = row[0].searchScore
        newRows.append(row)
    else:
      newRows.append(row)
      prevKey = (row[0].run, row[0].spectrumId)
  return newRows

if __name__ == "__main__":
   main()
