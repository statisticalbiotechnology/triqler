from __future__ import print_function

import collections

import numpy as np

from .. import parsers
from . import normalize_intensities as normalize
from . import percolator

def parsePsmsPoutFiles(psmsOutputFiles, key = lambda psm : psm.scannr):
  specToPeptideMap = collections.defaultdict(list)
  for psmsOutputFile in psmsOutputFiles:
    for psm in percolator.parsePsmsPout(psmsOutputFile):
      specToPeptideMap[key(psm)] = (psm.peptide, psm.proteins, psm.svm_score, psm.charge)
  return lambda spectrumIdx : specToPeptideMap.get(spectrumIdx, getDefaultPeptideHit())

def getDefaultPeptideHit():
  return ("NA", ["NA"], np.nan, -1) # psm.peptide, psm.PEP, psm.proteins, psm.svm_score, psm.charge
  
def getNormalizationFactorArrays(peptideToFeatureMap, fileInfoList, params):
  _, _, sampleList, fractionList = zip(*fileInfoList)
  rTimeArrays, factorArrays = dict(), dict()
  if not params['skipNormalization']:
    print("Applying retention-time dependent intensity normalization")
    minRunsObservedIn = len(set(sampleList)) / 3 + 1
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
  
  return rTimeArrays, factorArrays

def writeTriqlerInputFile(triqlerInputFile, peptideToFeatureMap, rTimeArrays, factorArrays, params):
  writer = parsers.getTsvWriter(triqlerInputFile)
  if params['simpleOutputFormat']:
    writer.writerow(parsers.TriqlerSimpleInputRowHeaders)
  else:
    writer.writerow(parsers.TriqlerInputRowHeaders)
  
  for featureClusterIdx, featureCluster in enumerate(peptideToFeatureMap.values()):
    if featureClusterIdx % 50000 == 0:
      print("Processing feature group", featureClusterIdx + 1)
    newRows = selectBestScorePerRun(featureCluster)
    
    if not params['skipMBR']:
      searchScores = [x[0].searchScore for x in newRows if not np.isnan(x[0].searchScore)]
      if len(searchScores) == 0:
        continue # this can happen if the only PSM has a searchScore <= 0
      worstSearchScore = np.min(searchScores)
    
    for (row, rTime, fraction) in newRows:
      if not params['skipNormalization']:
        newIntensity = normalize.getNormalizedIntensity(rTimeArrays[fraction][row.run], factorArrays[fraction][row.run], rTime, row.intensity)
        row = row._replace(intensity = newIntensity)
      
      if np.isnan(row.searchScore):
        if not params['skipMBR']:
          row = row._replace(searchScore = worstSearchScore)
        else:
          continue
      
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

