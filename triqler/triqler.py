from __future__ import print_function

"""triqler.triqler: provides entry point main()."""

__version__ = "0.6.0"
__copyright__ = '''Copyright (c) 2018-2020 Matthew The. All rights reserved.
Written by Matthew The (matthew.the@scilifelab.se) in the
School of Engineering Sciences in Chemistry, Biotechnology and Health at the 
Royal Institute of Technology in Stockholm.'''

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
from . import pgm
from . import diff_exp

def main():
  print('Triqler version %s\n%s' % (__version__, __copyright__))
  print('Issued command:', os.path.basename(__file__) + " " + " ".join(map(str, sys.argv[1:])))
  
  args, params = parseArgs()
  
  params['warningFilter'] = "ignore"
  with warnings.catch_warnings():
    warnings.simplefilter(params['warningFilter'])
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
  
  apars.add_argument('--write_protein_posteriors', default = '', metavar='P_OUT',
                     help='Write raw data of protein posteriors to the specified file in TSV format.')
  
  apars.add_argument('--write_group_posteriors', default = '', metavar='G_OUT',
                     help='Write raw data of treatment group posteriors to the specified file in TSV format.')
  
  apars.add_argument('--write_fold_change_posteriors', default = '', metavar='F_OUT',
                     help='Write raw data of fold change posteriors to the specified file in TSV format.')
  
  # ------------------------------------------------
  args = apars.parse_args()
  
  params = dict()
  params['warningFilter'] = "default"
  params['foldChangeEval'] = args.fold_change_eval
  params['t-test'] = args.ttest
  params['minSamples'] = args.min_samples
  params['decoyPattern'] = args.decoy_pattern
  params['numThreads'] = args.num_threads
  params['writeSpectrumQuants'] = args.write_spectrum_quants
  params['proteinPosteriorsOutput'] = args.write_protein_posteriors
  params['groupPosteriorsOutput'] = args.write_group_posteriors
  params['foldChangePosteriorsOutput'] = args.write_fold_change_posteriors
  params['returnPosteriors'] = len(params['proteinPosteriorsOutput']) > 0 or len(params['groupPosteriorsOutput']) > 0 or len(params['foldChangePosteriorsOutput']) > 0
  
  if params['minSamples'] < 2:
    sys.exit("ERROR: --min_samples should be >= 2")
  
  return args, params
  
def runTriqler(params, triqlerInputFile, triqlerOutputFile):  
  from timeit import default_timer as timer

  start = timer()

  if not os.path.isfile(triqlerInputFile):
    sys.exit("Could not locate input file %s. Check if the path is correct." % triqlerInputFile)
  
  params['hasLinkPEPs'] = parsers.hasLinkPEPs(triqlerInputFile)
  if triqlerInputFile.endswith(".pqr.tsv"):
    params['fileList'], params['groups'], params['groupLabels'], peptQuantRows = parsers.parsePeptideQuantFile(triqlerInputFile)
  else:
    peptQuantRowFile = triqlerInputFile + ".pqr.tsv"
    peptQuantRows = convertTriqlerInputToPeptQuantRows(triqlerInputFile, peptQuantRowFile, params)
  
  qvalMethod = 'pvalues' if params['t-test'] else 'avg_pep'
  
  selectComparisonBayesTmp = lambda proteinOutputRows, comparisonKey : selectComparisonBayes(proteinOutputRows, comparisonKey, params['t-test'])
  diff_exp.doDiffExp(params, peptQuantRows, triqlerOutputFile, doPickedProteinQuantification, selectComparisonBayesTmp, qvalMethod = qvalMethod)

  end = timer()
  print("Triqler execution took", end - start, "seconds wall clock time")

def convertTriqlerInputToPeptQuantRows(triqlerInputFile, peptQuantRowFile, params):
  peptQuantRowMap, getPEPFromScore, params['fileList'], params['groupLabels'], params['groups'] = groupTriqlerRowsByFeatureGroup(triqlerInputFile, params['decoyPattern'])
  
  if params['hasLinkPEPs'] and params['writeSpectrumQuants']:
    _, spectrumQuantRows, intensityDiv = _selectBestFeaturesPerRunAndPeptide(
        peptQuantRowMap, getPEPFromScore, params, 
        groupingKey = lambda x : x.spectrumId)
    spectrumQuantRows = _divideIntensities(spectrumQuantRows, intensityDiv)
    spectrumQuantRows = _updateIdentPEPs(spectrumQuantRows, params['decoyPattern'], params['hasLinkPEPs'])
    
    specQuantRowFile = triqlerInputFile + ".sqr.tsv"
    print("Writing spectrum quant rows to file:", specQuantRowFile)
    parsers.printPeptideQuantRows(specQuantRowFile, parsers.getRunIds(params), spectrumQuantRows)
  
  spectrumToFeatureMatch, peptideQuantRows, intensityDiv = _selectBestFeaturesPerRunAndPeptide(peptQuantRowMap, getPEPFromScore, params)
  peptideQuantRows = _selectBestPeptideQuantRowPerFeatureGroup(spectrumToFeatureMatch, peptideQuantRows)
  peptideQuantRows = _divideIntensities(peptideQuantRows, intensityDiv)
  peptideQuantRows = _updateIdentPEPs(peptideQuantRows, params['decoyPattern'], params['hasLinkPEPs'])
  
  print("Writing peptide quant rows to file:", peptQuantRowFile)
  parsers.printPeptideQuantRows(peptQuantRowFile, parsers.getRunIds(params), peptideQuantRows)
  
  return peptideQuantRows

def groupTriqlerRowsByFeatureGroup(triqlerInputFile, decoyPattern):
  print("Parsing triqler input file")
  
  peptQuantRowMap = collections.defaultdict(list)
  seenSpectra = set()
  targetScores, decoyScores = list(), list()
  runCondPairs = list()
  for i, trqRow in enumerate(parsers.parseTriqlerInputFile(triqlerInputFile)):
    if i % 1000000 == 0:
      print("  Reading row", i)
    
    peptQuantRowMap[trqRow.featureClusterId].append(trqRow)
    if (trqRow.run, trqRow.condition) not in runCondPairs:
      runCondPairs.append((trqRow.run, trqRow.condition))
    
    if not np.isnan(trqRow.searchScore) and trqRow.spectrumId not in seenSpectra:
      if _isDecoy(trqRow.proteins, decoyPattern):
        decoyScores.append(trqRow.searchScore)
      else:
        targetScores.append(trqRow.searchScore)
      seenSpectra.add(trqRow.spectrumId)
  
  fileList, groupLabels, groups = _getFilesAndGroups(runCondPairs)
  
  print("Calculating identification PEPs")
  getPEPFromScore = qvality.getPEPFromScoreLambda(targetScores, decoyScores)
    
  return peptQuantRowMap, getPEPFromScore, fileList, groupLabels, groups

def _selectBestFeaturesPerRunAndPeptide(peptQuantRowMap, getPEPFromScore, params, groupingKey = lambda x : x.peptide):
  print("Selecting best feature per run and spectrum")
  numRuns = len(params['fileList'])
  
  minIntensity = 1e100
  noSpectrum = 0
  peptideQuantRows = list()
  spectrumToFeatureMatch = dict() # stores the best peptideQuantRow per (protein, spectrumIdx)-pair
  for featureGroupIdx, trqRows in peptQuantRowMap.items():
    if featureGroupIdx % 100000 == 0:
      print("  featureGroupIdx:", featureGroupIdx)
    
    bestFeaturesPerRun = _selectBestFeaturesPerFeatureGroup(trqRows, 
        getPEPFromScore, params['fileList'], groupingKey, numRuns)
    
    for gKey in bestFeaturesPerRun:
      numRunsPresent = sum(1 for x in bestFeaturesPerRun[gKey] if x[0] < 1.01)
      if numRunsPresent < params['minSamples']:
        continue
      
      pqr = _convertFeatureGroupToPeptideQuantRow(bestFeaturesPerRun[gKey], 
          getPEPFromScore, featureGroupIdx, numRuns)
      
       # some feature clusters might not have a spectrum associated with them
      if pqr.spectrum == 0:
        noSpectrum += 1
        pqr = pqr._replace(spectrum = -100 * noSpectrum)
      
      peptideQuantRows.append(pqr)
      
      minIntensity = min(minIntensity, min([x for x in pqr.quant if x > 0.0]))
      
      # combinedPEP field temporarily contains SVM score
      identPEP = getPEPFromScore(pqr.combinedPEP)
      peptLinkErrorProb = 1.0 - np.prod([1.0 - x for x in pqr.linkPEP if x < 1.01])
      combinedPEP = _combinePEPs(identPEP, peptLinkErrorProb)
      
      # multiple featureGroups can be associated with the same consensus spectrum
      # when two or more analytes match closely in prec m/z and retention time;
      # choose the best featureGroup per (peptide, spectrum)-pair based on combinedPEP
      # note that chimeric spectra can still be associated with multiple peptideQuantRows, 
      # as the protein is included in the key
      key = (",".join(pqr.protein), pqr.spectrum / 100)
      if combinedPEP < spectrumToFeatureMatch.get(key, (-1, -1, 1.01))[2]:
        spectrumToFeatureMatch[key] = (pqr.spectrum, featureGroupIdx, combinedPEP)
  
  # divide intensities by a power of 10 for increased readability of peptide 
  # output file, make sure that the lowest intensity retains two significant 
  # digits after printing with two digits after the decimal point
  intensityDiv = np.power(10, np.floor(np.log10(minIntensity))+1)
  
  return spectrumToFeatureMatch, peptideQuantRows, intensityDiv

def _selectBestFeaturesPerFeatureGroup(trqRows, getPEPFromScore, fileList, groupingKey, numRuns):
  # groupingKey => array([linkPEP, triqlerInputRow])
  bestFeaturesPerRun = collections.defaultdict(lambda : [(1.01, None)]*numRuns)
  
  for trqRow in trqRows:
    fileIdx = fileList.index(trqRow.run)
    gKey = groupingKey(trqRow)      
    bestPEPForRun, bestTrqRowForRun = bestFeaturesPerRun[gKey][fileIdx]

    combinedPEP = _combinePEPs(trqRow.linkPEP, getPEPFromScore(trqRow.searchScore))
    
    samePEPhigherIntensity = (combinedPEP == bestPEPForRun and 
        trqRow.intensity > bestTrqRowForRun.intensity)
    if combinedPEP < bestPEPForRun or samePEPhigherIntensity:
      bestFeaturesPerRun[gKey][fileIdx] = (combinedPEP, trqRow)
  
  return bestFeaturesPerRun

def _convertFeatureGroupToPeptideQuantRow(bestFeaturesPerRun, getPEPFromScore, 
    featureGroupIdx, numRuns):
  intensities, linkPEPs, identPEPs = [0.0]*numRuns, [1.01]*numRuns, [1.01]*numRuns
  first = True
  svmScore = -1e9
  for fileIdx, (_, trqRow) in enumerate(bestFeaturesPerRun):
    if trqRow == None:
      continue
    
    if first:
      charge, proteins, first = trqRow.charge, trqRow.proteins, False
    
    if trqRow.searchScore > svmScore or np.isnan(trqRow.searchScore):
      svmScore, spectrumId, peptide = trqRow.searchScore, trqRow.spectrumId, trqRow.peptide
    
    intensities[fileIdx] = trqRow.intensity
    linkPEPs[fileIdx] = trqRow.linkPEP #_combinePEPs(linkPEP, identPEP)
    identPEPs[fileIdx] = getPEPFromScore(trqRow.searchScore)
  
  # fill in PEPs for missing values
  linkPEPs = _setMissingAsMax(linkPEPs)
  identPEPs = _setMissingAsMax(identPEPs)
  
  return parsers.PeptideQuantRow(svmScore, charge, featureGroupIdx, 
      spectrumId, linkPEPs, intensities, identPEPs, peptide, proteins)
    
def _setMissingAsMax(PEPs):
  maxPEP = max([x for x in PEPs if x <= 1.0])
  return [x if x <= 1.0 else maxPEP for x in PEPs]
  
def _getFilesAndGroups(runCondPairs):
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
  
  if len(fileList) < 2:
    sys.exit("ERROR: There should be at least two runs.")
  elif len(groups) < 2:
    sys.exit("ERROR: At least two conditions (treatment groups) should be specified.")
  elif min([len(g) for g in groups]) < 2:
    sys.exit("ERROR: Each condition (treatment group) should have at least two runs.")
  
  return fileList, groupLabels, groups

# multiple peptideQuantRows can be associated with the same featureGroup 
# when two or more analytes match closely in prec m/z and retention time;
# choose the best peptideQuantRow per featureGroup based on combinedPEP
def _selectBestPeptideQuantRowPerFeatureGroup(spectrumToFeatureMatch, peptideQuantRows):
  featureGroupToSpectrumIdx = dict()
  for (spectrumIdx, featureGroupIdx, combinedPEP) in spectrumToFeatureMatch.values():
    if combinedPEP < featureGroupToSpectrumIdx.get(featureGroupIdx, (-1, 1.01))[1]:
      featureGroupToSpectrumIdx[featureGroupIdx] = (spectrumIdx, combinedPEP)
  survivingSpectrumIdxs = set([y[0] for x, y in featureGroupToSpectrumIdx.items()])
  #print("Surviving spectrumIdxs:", len(survivingSpectrumIdxs))
  
  peptideQuantRows = filter(lambda x : x.spectrum in survivingSpectrumIdxs, peptideQuantRows)
  
  return peptideQuantRows

def _divideIntensities(peptideQuantRows, intensityDiv = 1e6):
  print("Dividing intensities by %g for increased readability" % intensityDiv)
  newPeptideQuantRows = list()
  for row in peptideQuantRows:
    row = row._replace(quant = list(map(lambda x : x/intensityDiv, row.quant)))
    newPeptideQuantRows.append(row)
  return newPeptideQuantRows

def doPickedProteinQuantification(peptQuantRows, params, proteinModifier, getEvalFeatures):
  notPickedProteinOutputRows = _groupPeptideQuantRowsByProtein(
      peptQuantRows, proteinModifier, params['decoyPattern'])
  
  np.random.shuffle(notPickedProteinOutputRows)
  notPickedProteinOutputRows = sorted(notPickedProteinOutputRows, key = lambda x : x[0], reverse = True)
  
  print("Calculating protein-level identification PEPs")
  pickedProteinOutputRows, proteinPEPs = _pickedProteinStrategy(notPickedProteinOutputRows, params['decoyPattern'])  
  
  print("Fitting hyperparameters")
  hyperparameters.fitPriors(peptQuantRows, params)
  
  print("Calculating protein posteriors")
  posteriors = getPosteriors(pickedProteinOutputRows, proteinPEPs, params)
  
  proteinQuantRows = _updateProteinQuantRows(pickedProteinOutputRows, posteriors, proteinPEPs, getEvalFeatures, params)
  
  return proteinQuantRows

def _updateProteinQuantRows(pickedProteinOutputRows, posteriors, proteinPEPs, 
    getEvalFeatures, params):
  proteinQuantRows = list()
  sumPEP = 0.0
  for (linkPEP, protein, quantRows, numPeptides), (bayesQuantRow, muGroupDiffs, probsBelowFoldChange, posteriorDists), proteinPEP in zip(pickedProteinOutputRows, posteriors, proteinPEPs):
    evalFeatures = getEvalFeatures(bayesQuantRow)
    
    if not params['t-test']:
      evalFeatures[-1] = probsBelowFoldChange
      evalFeatures[-2] = muGroupDiffs
    
    sumPEP += proteinPEP
    if not params['t-test'] or sumPEP / (len(proteinQuantRows) + 1) < 0.05:
      proteinQuantRows.append([linkPEP, protein, quantRows, evalFeatures, numPeptides, proteinPEP, bayesQuantRow, posteriorDists])
    
  proteinQuantRows = sorted(proteinQuantRows, key = lambda x : (x[0], x[1]))
  return proteinQuantRows

def _groupPeptideQuantRowsByProtein(peptQuantRows, proteinModifier, decoyPattern):
  protQuantRows = parsers.filterAndGroupPeptides(peptQuantRows)
  
  proteinRows = list()
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
    protein = proteinModifier(prot)
    numPeptides = usablePeptides

    proteinOutputRow = (_getProteinScore(quantRows), list(quantRows[psmIdx].linkPEP), protein, quantRows, numPeptides)
    proteinRows.append(proteinOutputRow)
  
  return proteinRows

def _getProteinScore(quantRows):
  # logged version performs slightly worse on iPRG2016 set, but might 
  # prevent convergence problems in the case of many peptides for a protein
  return np.log(-1*np.sum([np.log(x.combinedPEP) for x in quantRows]))

def _pickedProteinStrategy(notPickedProteinOutputRows, decoyPattern):
  pickedProteinOutputRows = list()
  targetScores, decoyScores = list(), list()
  seenProteins = set()
  for score, linkPEP, protein, quantRows, numPeptides in notPickedProteinOutputRows:
    evalProtein = protein.replace(decoyPattern, "", 1)
    if evalProtein not in seenProteins:
      seenProteins.add(evalProtein)
      if _isDecoy([protein], decoyPattern):
        decoyScores.append(score)
      else:
        targetScores.append(score)
      pickedProteinOutputRows.append([linkPEP, protein, quantRows, numPeptides])
  
  targetScores = np.array(targetScores)
  decoyScores = np.array(decoyScores)
  _, proteinPEPs = qvality.getQvaluesFromScores(targetScores, decoyScores, includePEPs = True, includeDecoys = True, tdcInput = True)
  
  if len(np.nonzero(proteinPEPs < 1.0)) == 0:
    sys.exit("ERROR: No proteins could be identified with a PEP below 1.0, cannot calculate posteriors.")
  else:
    print("  Identified", qvality.countBelowFDR(proteinPEPs, 0.01), "proteins at 1% FDR")
  
  return pickedProteinOutputRows, proteinPEPs

def getPosteriors(pickedProteinOutputRows, peps, params):
  if params['numThreads'] > 1:
    from . import multiprocessing_pool as pool
    processingPool = pool.MyPool(processes = params['numThreads'], warningFilter = params['warningFilter'])
  
  addDummyPosteriors = 0
  posteriors = list()
  for (linkPEP, protein, quantRows, numPeptides), proteinIdPEP in zip(pickedProteinOutputRows, peps):  
    if proteinIdPEP < 1.0:
      if params['numThreads'] > 1:
        processingPool.applyAsync(pgm.getPosteriors, [quantRows, params])
      else:
        posteriors.append(pgm.getPosteriors(quantRows, params))
        if len(posteriors) % 50 == 0:
          print(" ", len(posteriors),"/", sum(1 for p in peps if p < 1.0), "%.2f" % (float(len(posteriors)) / sum(1 for p in peps if p < 1.0) * 100) + "%")
    else:
      addDummyPosteriors += 1
  
  if params['numThreads'] > 1:
    posteriors = processingPool.checkPool(printProgressEvery = 50)
  
  posteriors.extend([pgm.getDummyPosteriors(params)] * addDummyPosteriors)
  
  return posteriors
  
def selectComparisonBayes(proteinOutputRows, comparisonKey, tTest = False):
  proteinOutputRowsUpdatedPEP = list()
  for (linkPEP, protein, quantRows, evalFeatures, numPeptides, proteinPEP, bayesQuantRow, posteriorDists) in proteinOutputRows:
    evalFeaturesNew = copy.deepcopy(evalFeatures)
    evalFeaturesNew[-1] = evalFeatures[-1][comparisonKey] # probBelowFoldChange
    evalFeaturesNew[-2] = evalFeatures[-2][comparisonKey] # log2_fold_change
    if not tTest:
      combinedPEP = _combinePEPs(evalFeaturesNew[-1], proteinPEP)
    else:
      combinedPEP = evalFeaturesNew[-1]
    
    proteinOutputRowsUpdatedPEP.append([combinedPEP, linkPEP, protein, quantRows, evalFeaturesNew, numPeptides, proteinPEP, bayesQuantRow, posteriorDists])

  proteinOutputRowsUpdatedPEP = sorted(proteinOutputRowsUpdatedPEP, key = lambda x : (x[0], x[1]))
  return proteinOutputRowsUpdatedPEP

# calculate peptide-level identification FDRs and update the linkPEPs with this estimate
def _updateIdentPEPs(peptideQuantRows, decoyPattern, hasLinkPEPs):
  print("Calculating peptide-level identification PEPs")
  
  scoreIdxPairs = list()
  for i, row in enumerate(peptideQuantRows):
    if not np.isnan(row.combinedPEP):
      # row.combinedPEP contains the SVM score
      scoreIdxPairs.append([row.combinedPEP, i, _isDecoy(row.protein, decoyPattern)]) 
  
  scoreIdxPairs = sorted(scoreIdxPairs, reverse = True)
  scoreIdxs = np.argsort([x[1] for x in scoreIdxPairs])
  targetScores = np.array([x[0] for x in scoreIdxPairs if x[2] == False])
  decoyScores = np.array([x[0] for x in scoreIdxPairs if x[2] == True])
  
  _, identPEPs = qvality.getQvaluesFromScores(targetScores, decoyScores, includePEPs = True, includeDecoys = True, tdcInput = True)
  
  print("  Identified", qvality.countBelowFDR(identPEPs, 0.01), "peptides at 1% FDR")
  newPeptideQuantRows = list()
  i = 0
  for row in peptideQuantRows:
    identPEP = 1.0
    if not np.isnan(row.combinedPEP):
      identPEP = identPEPs[scoreIdxs[i]]
      i += 1
    
    if hasLinkPEPs:
      newPeptideQuantRows.append(row._replace(combinedPEP = identPEP)) # using consensus spectra
    else:
      newPeptideQuantRows.append(row._replace(combinedPEP = identPEP, identificationPEP = [_combinePEPs(identPEP, x) for x in row.identificationPEP]))
  return newPeptideQuantRows

def _isDecoy(proteins, decoyPattern):
  isDecoyProt = True
  for protein in proteins:
    if not protein.startswith(decoyPattern):
      isDecoyProt = False
      break
  return isDecoyProt
  
def _combinePEPs(linkPEP, identPEP):
  return 1.0 - (1.0 - linkPEP)*(1.0 - identPEP)

if __name__ == "__main__":
  main()
