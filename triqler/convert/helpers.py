from __future__ import print_function

import collections

import numpy as np

from .. import parsers
from . import normalize_intensities as normalize
from . import percolator

AA_MASSES = {
    'G': 57.02146,
    'A': 71.03711,
    'S': 87.03203,
    'P': 97.05276,
    'V': 99.06841,
    'T': 101.04768,
    'C': 103.00919+57.021464,
    'L': 113.08406,
    'I': 113.08406,
    'N': 114.04293,
    'D': 115.02694,
    'Q': 128.05858,
    'K': 128.09496,
    'E': 129.04259,
    'M': 131.04049,
    'H': 137.05891,
    'F': 147.06841,
    'U': 150.95364,
    'R': 156.10111,
    'Y': 163.06333,
    'W': 186.07931,
    'O': 237.14773,
}

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

def getMods(modPeptide):
  peptide = ""
  peptideIdx = 0
  mods = [0]
  inMod = False
  for i in range(len(modPeptide)):
    if modPeptide[i] == "[":
      j = modPeptide[i:].find("]")
      if modPeptide[i+1:i+j].startswith("UNIMOD:"):
        unimodId = int(modPeptide[i+1:i+j].split(":")[1])
        if unimodId == 4: # Carboxyamidomethylation peptide N-term or aa
          if i == 0 or modPeptide[i-1] != "C":
            massDiff = 57.021464
          else:
            massDiff = 0.0
        elif unimodId == 5: # Carbamylation peptide N-term
          massDiff = 43.005814
        elif unimodId == 1: # Acetylation peptide N-term
          massDiff = 42.010565
        elif unimodId == 28 or unimodId == 385: # Pyro-glu from Q / Pyro-carbamidomethyl as a delta from Carbamidomethyl-Cys
          massDiff = -17.026549
        elif unimodId == 27: # Pyro-glu from E
          massDiff = -18.010565
        elif unimodId == 35: # Oxidation of M
          massDiff = 15.994915
        else:
          sys.exit("Unknown UNIMOD id: " + str(unimodId))
        mods[peptideIdx] += massDiff
      else:
        mods[peptideIdx] += float(modPeptide[i+1:i+j])
      if mods[peptideIdx] == 16.0: # more accurate monoisotopic mass for oxidations
        mods[peptideIdx] = 15.994915
      inMod = True
    elif modPeptide[i] == "]":
      inMod = False
    elif not inMod:
      peptide += modPeptide[i]
      peptideIdx += 1
      mods.append(0)
  return mods, peptide
  
def calcMass(peptide):
  mods, peptide = getMods(peptide[2:-2])
  return sum(mods) + sum([AA_MASSES[p] for p in peptide]) + 18.010565 + 1.00727646677

def precMzFromPrecMass(pmass, z):
  return (float(pmass) + 1.00727646677 * (int(z) - 1)) / int(z)

def precMassFromPrecMz(pmz, z):
  return pmz * z - 1.00727646677 * (z - 1)
  
