from __future__ import print_function

"""triqler.triqler: provides entry point main()."""

__version__ = "0.1.2"

import sys
import os
import collections
import copy
import csv
import multiprocessing
import warnings

import numpy as np

from . import parsers
from . import qvality
from . import hyperparameters
from . import multiprocessing_pool as pool
from . import pgm
from . import diff_exp

def main():
  print('''Triqler version %s
Copyright (c) 2018 Matthew The. All rights reserved.
Written by Matthew The (matthew.the@scilifelab.se) in the
School of Engineering Sciences in Chemistry, Biotechnology and Health at the 
Royal Institute of Technology in Stockholm.
  ''' % (__version__))
  args, params = parseArgs()
  
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    runTriqler(params, args.in_file, args.out_file)

def parseArgs():
  import argparse
  apars = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  apars.add_argument('in_file', default=None, metavar = "IN_FILE",
                     help='''List of PSMs with abundances (not log transformed!) 
                             and search engine score. See README for a detailed 
                             description of the columns.
                          ''')
  
  apars.add_argument('--out_file', default = "proteins.tsv", metavar='OUT', 
                     help='''Path to output file (writing in TSV format). 
                             N.B. if more than 2 treatment groups are present, 
                             suffixes will be added before the file extension.
                          ''')
  
  apars.add_argument('--fold_change_eval', type=float, default=1.0, metavar='F',
                     help='log2 fold change evaluation threshold.')
                     
  apars.add_argument('--decoy_pattern', default = "decoy_", metavar='P', 
                     help='Prefix for decoy proteins.')
                     
  apars.add_argument('--min_samples', type=int, default=2, metavar='N', 
                     help='Minimum number of samples a peptide needed to be quantified in.')
  # Peptides quantified in less than the minimum number will be discarded
  
  apars.add_argument('--num_threads', type=int, default=multiprocessing.cpu_count(), metavar='N', 
                     help='Number of threads, by default this is equal to the number of CPU cores available on the device.')
  
  apars.add_argument('--ttest',
                     help='Use t-test for evaluating differential expression instead of posterior probabilities.',
                     action='store_true')
  
  apars.add_argument('--write_spectrum_quants',
                     help='Write quantifications for consensus spectra. Only works if consensus spectrum index are given in input.',
                     action='store_true')
                       
  # ------------------------------------------------
  args = apars.parse_args()
  
  params = dict()
  params['foldChangeEval'] = args.fold_change_eval
  params['t-test'] = args.ttest
  params['minSamples'] = args.min_samples
  params['decoyPattern'] = args.decoy_pattern
  params['numThreads'] = args.num_threads
  params['writeSpectrumQuants'] = args.write_spectrum_quants
  
  if params['minSamples'] < 2:
    sys.exit("ERROR: --min_samples should be >= 2")
  
  return args, params
  
def runTriqler(params, triqlerInputFile, triqlerOutputFile):  
  if not os.path.isfile(triqlerInputFile):
    sys.exit("Could not locate input file %s. Check if the path to the input file is correct." % triqlerInputFile)
  peptQuantRowFile = triqlerInputFile + ".pqr.tsv"
  peptQuantRows = convertTriqlerInputToPeptQuantRows(triqlerInputFile, peptQuantRowFile, params)
  if params['t-test']:
    qvalMethod = 'pvalues'
  else:
    qvalMethod = 'avg_pep'
    
  selectComparisonBayesTmp = lambda proteinOutputRows, comparisonKey : selectComparisonBayes(proteinOutputRows, comparisonKey, params['t-test'])
  diff_exp.doDiffExp(params, peptQuantRows, triqlerOutputFile, getPickedProteinCalibration, selectComparisonBayesTmp, qvalMethod = qvalMethod)

def convertTriqlerInputToPeptQuantRows(triqlerInputFile, peptQuantRowFile, params):
  peptQuantRowMap, getPEPFromScore, params['fileList'], params['groupLabels'], params['groups'], params['hasLinkPEPs'] = getPeptQuantRowMap(triqlerInputFile, params['decoyPattern'])
  
  if params['hasLinkPEPs'] and params['writeSpectrumQuants']:
    spectrumToFeatureMatch, featureClusterRows, intensityDiv = selectBestFeaturesPerRunAndSpectrum(peptQuantRowMap, getPEPFromScore, params, reduceKey = lambda x : x.spectrumId)
    
    peptideQuantRows = convertToPeptideQuantRows(featureClusterRows, intensityDiv)
    
    peptideQuantRows = updateIdentPEPs(peptideQuantRows, params['decoyPattern'], params['hasLinkPEPs'])
    
    print("Writing spectrum quant rows to file")
    specQuantRowFile = triqlerInputFile + ".sqr.tsv"
    printPeptideQuantRows(specQuantRowFile, parsers.getRunIds(params), peptideQuantRows)
  
  spectrumToFeatureMatch, featureClusterRows, intensityDiv = selectBestFeaturesPerRunAndSpectrum(peptQuantRowMap, getPEPFromScore, params)
  
  featureClusterRows = selectBestPqrPerFeatureCluster(spectrumToFeatureMatch, featureClusterRows)
  
  peptideQuantRows = convertToPeptideQuantRows(featureClusterRows, intensityDiv)
  
  peptideQuantRows = updateIdentPEPs(peptideQuantRows, params['decoyPattern'], params['hasLinkPEPs'])
  
  print("Writing peptide quant rows to file")
  printPeptideQuantRows(peptQuantRowFile, parsers.getRunIds(params), peptideQuantRows)
  
  return peptideQuantRows

def getPeptQuantRowMap(triqlerInputFile, decoyPattern):
  print("Parsing triqler input file")
  
  peptQuantRowMap = collections.defaultdict(list)
  seenSpectra = set()
  targetScores, decoyScores = list(), list()
  runCondPairs = list()
  for trqRow in parsers.parseTriqlerInputFile(triqlerInputFile):
    peptQuantRowMap[trqRow.featureClusterId].append(trqRow)
    if (trqRow.run, trqRow.condition) not in runCondPairs:
      runCondPairs.append((trqRow.run, trqRow.condition))
    
    if not np.isnan(trqRow.searchScore) and trqRow.spectrumId not in seenSpectra:
      if isDecoy(trqRow.proteins, decoyPattern):
        decoyScores.append(trqRow.searchScore)
      else:
        targetScores.append(trqRow.searchScore)
      seenSpectra.add(trqRow.spectrumId)
  
  if len(decoyScores) == 0:
    sys.exit("ERROR: No decoy hits found, check if the correct decoy prefix was specified with the --decoy_pattern flag")
  
  print("Calculating identification PEPs")
  
  targetScores = sorted(targetScores, reverse = True)
  decoyScores = sorted(decoyScores, reverse = True)
  _, peps = qvality.getQvaluesFromScores(targetScores, decoyScores, includePEPs = True, includeDecoys = True, tdcInput = True)
  peps = peps[::-1]
  
  allScores = np.array(sorted(targetScores + decoyScores))
  getPEPFromScore = lambda score : peps[min(np.searchsorted(allScores, score, side = 'left'), len(peps) - 1)] if not np.isnan(score) else 1.0
  
  fileList, groupLabels, groups = getFilesAndGroups(runCondPairs)
  
  if len(groups) < 2:
    sys.exit("ERROR: At least two treatment groups should be specified")
    
  return peptQuantRowMap, getPEPFromScore, fileList, groupLabels, groups, parsers.hasLinkPEPs(triqlerInputFile)

def setMissingAsMax(PEPs):
  maxPEP = max([x for x in PEPs if x <= 1.0])
  return [x if x <= 1.0 else maxPEP for x in PEPs]
      
def selectBestFeaturesPerRunAndSpectrum(peptQuantRowMap, getPEPFromScore, params, reduceKey = lambda x : x.peptide):
  minIntensity = 1e100
  noSpectrum = 0
  featureClusterRows = list()
  spectrumToFeatureMatch = dict() # stores the best peptideQuantRow per (protein, spectrumIdx)-pair
  for featureClusterIdx, trqRows in peptQuantRowMap.items():
    if featureClusterIdx % 10000 == 0:
      print("featureClusterIdx:", featureClusterIdx)

    bestPEP = collections.defaultdict(lambda : [(1.01, None)]*len(params['fileList'])) # peptide => array([linkPEP, precursorCandidate])
    bestPeptideScore = collections.defaultdict(lambda : (-1e9, -1)) # peptide => (identPEP, spectrumIdx); contains the identification PEPs
    
    for trqRow in trqRows:
      fileIdx = params['fileList'].index(trqRow.run)
      combinedPEP = combinePEPs(trqRow.linkPEP, getPEPFromScore(trqRow.searchScore))
      rKey = reduceKey(trqRow)
      if combinedPEP < bestPEP[rKey][fileIdx][0] or (combinedPEP == bestPEP[rKey][fileIdx][0] and trqRow.intensity > bestPEP[rKey][fileIdx][1].intensity):
        bestPEP[rKey][fileIdx] = (combinedPEP, trqRow)
        if trqRow.searchScore > bestPeptideScore[rKey][0] or np.isnan(trqRow.searchScore):
          bestPeptideScore[rKey] = (trqRow.searchScore, trqRow.spectrumId)
    
    for peptide in bestPEP:
      if sum(1 for x in bestPEP[peptide] if x[0] < 1.01) < params['minSamples']:
        continue
      intensities = [0.0]*len(params['fileList'])
      linkPEPs = [1.01]*len(params['fileList'])
      identPEPs = [1.01]*len(params['fileList'])
      peptLinkEP = 1.0
      first = True
      #svmScore, spectrumId = bestPeptideScore[peptide]
      #identPEP = getPEPFromScore(svmScore)
      for fileIdx, (_, trqRow) in enumerate(bestPEP[peptide]):
        if trqRow != None:
          if first:
            charge, proteins, first = trqRow.charge, trqRow.proteins, False
          identPEPs[fileIdx] = getPEPFromScore(trqRow.searchScore)
          peptLinkEP *= 1.0 - trqRow.linkPEP
          linkPEPs[fileIdx] = trqRow.linkPEP #combinePEPs(linkPEP, identPEP)
          intensities[fileIdx] = trqRow.intensity
      
      linkPEPs = setMissingAsMax(linkPEPs)
      identPEPs = setMissingAsMax(identPEPs)
      
      minIntensity = min(minIntensity, min([x for x in intensities if x > 0.0]))
      
      svmScore, spectrumId = bestPeptideScore[peptide]
      
      # some feature clusters might not have a spectrum associated with them
      if spectrumId == 0:
        noSpectrum += 1
        spectrumId = -100 * noSpectrum
      
      featureClusterRows.append((intensities, featureClusterIdx, spectrumId, linkPEPs, identPEPs, peptide, proteins, svmScore, charge))
      
      identPEP = getPEPFromScore(svmScore)
      combinedPEP = combinePEPs(identPEP, 1.0 - peptLinkEP)
      
      # multiple featureClusters can be associated with the same consensus spectrum
      # when two or more analytes match closely in prec m/z and retention time;
      # choose the best featureCluster per (peptide, spectrum)-pair based on combinedPEP
      # note that chimeric spectra can still be associated with multiple peptideQuantRows, 
      # as the protein is included in the key
      key = (",".join(proteins), spectrumId / 100)
      if combinedPEP < spectrumToFeatureMatch.get(key, (-1, -1, 1.01))[2]:
        spectrumToFeatureMatch[key] = (spectrumId, featureClusterIdx, combinedPEP)
  
  # divide intensities by a power of 10 for increased readability of peptide 
  # output file, make sure that the lowest intensity retains two significant 
  # digits after printing with two digits after the decimal point
  intensityDiv = np.power(10, np.floor(np.log10(minIntensity))+1)
  print("Dividing intensities by %g for increased readability" % intensityDiv)
  
  return spectrumToFeatureMatch, featureClusterRows, intensityDiv
  
def getFilesAndGroups(runCondPairs):
  runCondPairs = sorted(runCondPairs, key = lambda x : (x[1], x[0]))
  fileList = list()
  groupLabels, groups = list(), list()
  for run, cond in runCondPairs:
    if run not in fileList:
      fileList.append(run)
      if cond not in groupLabels:
        groupLabels.append(cond)
        groups.append([])
      groups[groupLabels.index(cond)].append(len(fileList) - 1)
  
  return fileList, groupLabels, groups

# multiple peptideQuantRows can be associated with the same featureCluster 
# when two or more analytes match closely in prec m/z and retention time;
# choose the best peptideQuantRow per featureCluster based on combinedPEP
def selectBestPqrPerFeatureCluster(spectrumToFeatureMatch, featureClusterRows):
  featureClusterToSpectrumIdx = dict()
  for (spectrumIdx, featureClusterIdx, combinedPEP) in spectrumToFeatureMatch.values():
    if combinedPEP < featureClusterToSpectrumIdx.get(featureClusterIdx, (-1, 1.01))[1]:
      featureClusterToSpectrumIdx[featureClusterIdx] = (spectrumIdx, combinedPEP)
  survivingSpectrumIdxs = set([y[0] for x, y in featureClusterToSpectrumIdx.items()])
  print("Surviving spectrumIdxs:", len(survivingSpectrumIdxs))
  
  featureClusterRows = filter(lambda x : x[2] in survivingSpectrumIdxs, featureClusterRows)
  
  return featureClusterRows

def convertToPeptideQuantRows(featureClusterRows, intensityDiv = 1e6):
  print("Converting to peptide quant rows")
  peptideQuantRows = list()
  for intensities, featureClusterIdx, spectrumIdx, linkPEPs, identPEPs, peptide, proteins, svmScore, charge in featureClusterRows:
    row = parsers.PeptideQuantRow(svmScore, charge, featureClusterIdx, linkPEPs, list(map(lambda x : x/intensityDiv, intensities)), identPEPs, peptide, proteins)
    peptideQuantRows.append(row)
  return peptideQuantRows
  
def printPeptideQuantRows(peptOutputFile, headers, peptideQuantRows):
  writer = csv.writer(open(peptOutputFile, 'w'), delimiter = '\t')
  writer.writerow(parsers.getPeptideQuantRowHeaders(headers))
  for row in peptideQuantRows:
    writer.writerow(row.toList())
     
def getProteinCalibration(peptQuantRows, proteinModifier, decoyPattern):
  protQuantRows = parsers.filterAndGroupPeptides(peptQuantRows)
  
  proteinTargetRows, proteinDecoyRows = list(), list()
  for prot, quantRows in protQuantRows:
    psmIdx = 0
    quantRows = sorted(quantRows, key = lambda x : x.combinedPEP)
    seenPeptides = set()
    usablePeptides = 0
    filteredQuantRows = list()
    for quantRow in quantRows:
      cleanPeptide = parsers.cleanPeptide(quantRow.peptide)
      if cleanPeptide not in seenPeptides:
        seenPeptides.add(cleanPeptide)
        usablePeptides += 1
        filteredQuantRows.append(quantRow)

    if usablePeptides < 1:
      continue
    else:
      quantRows = filteredQuantRows

    isDecoyProt = isDecoy([prot], decoyPattern)
    protein = proteinModifier(prot)
    numPeptides = usablePeptides

    proteinOutputRow = (quantRows[psmIdx].linkPEP, protein, quantRows, numPeptides)
    if isDecoyProt:
      proteinDecoyRows.append(proteinOutputRow)
    else:
      proteinTargetRows.append(proteinOutputRow)

  proteinTargetRows.sort()
  proteinDecoyRows.sort()
  return proteinTargetRows, proteinDecoyRows

def getPickedProteinCalibration(peptQuantRows, params, proteinModifier, getEvalFeatures):
  targetProteinOutputRows, decoyProteinOutputRows = getProteinCalibration(peptQuantRows, proteinModifier, params['decoyPattern'])

  pickedProteinOutputRows = targetProteinOutputRows + decoyProteinOutputRows
  np.random.shuffle(pickedProteinOutputRows)

  proteinQuantIdPEP = lambda quantRows : np.sum([np.log(x.combinedPEP) for x in quantRows]) # combinedPEP contains the peptide-level PEP
  pickedProteinOutputRows = sorted(pickedProteinOutputRows, key = lambda x : proteinQuantIdPEP(x[2]))
  
  print("Fitting hyperparameters")
  hyperparameters.fitPriors(peptQuantRows, params) # updates priors
  
  targetScores, decoyScores = list(), list()
  proteinOutputRows = list()
  seenProteins = set()
  
  print("Calculating protein quants")
  processingPool = pool.MyPool(params['numThreads'])
  pickedProteinOutputRowsNew = list()
  for linkPEP, protein, quantRows, numPeptides in pickedProteinOutputRows:
    evalProtein = protein.replace(params['decoyPattern'], "", 1)
    if evalProtein not in seenProteins:
      seenProteins.add(evalProtein)
      
      #score = np.log(-1*proteinQuantIdPEP(quantRows)) # performs slightly worse on iPRG2016 set, but might prevent convergence problems in the event of many peptides for a protein
      score = -1*proteinQuantIdPEP(quantRows)
      if isDecoy([protein], params['decoyPattern']):
        decoyScores.append(score)
      else:
        targetScores.append(score)
      pickedProteinOutputRowsNew.append([linkPEP, protein, quantRows, numPeptides])
      processingPool.applyAsync(pgm.getPosteriors, [quantRows, params])
      #pgm.getPosteriors(quantRows, params) # for debug mode
  posteriors = processingPool.checkPool(printProgressEvery = 50)
  
  print("Calculating protein-level identification PEPs")
  _, peps = qvality.getQvaluesFromScores(targetScores, decoyScores, includePEPs = True, includeDecoys = True, tdcInput = True)
  
  proteinOutputRowsUpdatedPEP = list()
  sumPEP = 0.0
  for (linkPEP, protein, quantRows, numPeptides), (bayesQuantRow, muGroupDiffs, probsBelowFoldChange), proteinPEP in zip(pickedProteinOutputRowsNew, posteriors, peps):
    evalFeatures = getEvalFeatures(bayesQuantRow)
    if not params['t-test']:
      evalFeatures[-1] = probsBelowFoldChange
      evalFeatures[-2] = muGroupDiffs
    
    if not params['t-test'] or sumPEP / (len(proteinOutputRowsUpdatedPEP) + 1) < 0.05:
      proteinOutputRowsUpdatedPEP.append([linkPEP, protein, quantRows, evalFeatures, numPeptides, proteinPEP, bayesQuantRow])
      sumPEP += proteinPEP
  
  proteinOutputRowsUpdatedPEP = sorted(proteinOutputRowsUpdatedPEP, key = lambda x : (x[0], x[1]))
  return proteinOutputRowsUpdatedPEP

def selectComparisonBayes(proteinOutputRows, comparisonKey, tTest = False):
  proteinOutputRowsUpdatedPEP = list()
  for (linkPEP, protein, quantRows, evalFeatures, numPeptides, proteinPEP, bayesQuantRow) in proteinOutputRows:
    evalFeaturesNew = copy.deepcopy(evalFeatures)
    evalFeaturesNew[-1] = evalFeatures[-1][comparisonKey] # probBelowFoldChange
    evalFeaturesNew[-2] = evalFeatures[-2][comparisonKey] # log2_fold_change
    if not tTest:
      combinedPEP = combinePEPs(evalFeaturesNew[-1], proteinPEP)
    else:
      combinedPEP = evalFeaturesNew[-1]
    
    proteinOutputRowsUpdatedPEP.append([combinedPEP, linkPEP, protein, quantRows, evalFeaturesNew, numPeptides, proteinPEP, bayesQuantRow])

  proteinOutputRowsUpdatedPEP = sorted(proteinOutputRowsUpdatedPEP, key = lambda x : (x[0], x[1]))
  return proteinOutputRowsUpdatedPEP

# calculate peptide-level identification FDRs and update the linkPEPs with this estimate
def updateIdentPEPs(peptideQuantRow, decoyPattern, hasLinkPEPs):
  print("Calculating peptide-level identification PEPs")
  
  scoreIdxPairs = list()
  for i, row in enumerate(peptideQuantRow):
    if not np.isnan(row.combinedPEP):
      scoreIdxPairs.append([row.combinedPEP, i, isDecoy(row.protein, decoyPattern)]) # row.combinedPEP contains the SVM score
  
  scoreIdxPairs = sorted(scoreIdxPairs, reverse = True)
  scoreIdxs = np.argsort([x[1] for x in scoreIdxPairs])
  targetScores = [x[0] for x in scoreIdxPairs if x[2] == False]
  decoyScores = [x[0] for x in scoreIdxPairs if x[2] == True]
  
  _, identPEPs = qvality.getQvaluesFromScores(targetScores, decoyScores, includePEPs = True, includeDecoys = True, tdcInput = True)
  
  newPeptideQuantRows = list()
  i = 0
  for row in peptideQuantRow:
    if not np.isnan(row.combinedPEP):
      identPEP = identPEPs[scoreIdxs[i]]
      i += 1
    else:
      identPEP = 1.0
    
    if hasLinkPEPs:
      newPeptideQuantRows.append(row._replace(combinedPEP = identPEP)) # using consensus spectra
    else:
      newPeptideQuantRows.append(row._replace(combinedPEP = identPEP, identificationPEP = [combinePEPs(identPEP, x) for x in row.identificationPEP]))
  return newPeptideQuantRows
   
def isDecoy(proteins, decoyPattern):
  isDecoyProt = True
  for protein in proteins:
    if not protein.startswith(decoyPattern):
      isDecoyProt = False
      break
  return isDecoyProt
  
def combinePEPs(linkPEP, identPEP):
  return 1.0 - (1.0 - linkPEP)*(1.0 - identPEP)

def scoreFromPEP(PEP):
  return -1*np.log(PEP + 1e-18)

if __name__ == "__main__":
  main()
