from __future__ import print_function

"""triqler.triqler: provides entry point main()."""

__version__ = "0.1.1"

import sys
import random
import collections
import getopt
import itertools
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
                     
  apars.add_argument('--min_samples', type=int, default=1, metavar='N', 
                     help='Minimum number of samples a peptide needed to be quantified in.')
  # Peptides quantified in less than the minimum number will be discarded
  
  apars.add_argument('--num_threads', type=int, default=multiprocessing.cpu_count(), metavar='N', 
                     help='Number of threads, by default this is equal to the number of CPU cores available on the device.')
  # Peptides quantified in less than the minimum number will be discarded
  
  apars.add_argument('--ttest',
                     help='Use t-test for evaluating differential expression instead of posterior probabilities.',
                     action='store_true')
                       
  # ------------------------------------------------
  args = apars.parse_args()
  
  params = dict()
  params['foldChangeEval'] = args.fold_change_eval
  params['t-test'] = args.ttest
  params['minSamples'] = args.min_samples
  params['decoyPattern'] = args.decoy_pattern
  params['numThreads'] = args.num_threads
  
  return args, params
  
def runTriqler(params, triqlerInputFile, triqlerOutputFile):  
  peptQuantRowFile = triqlerInputFile + ".pqr.tsv"
  peptQuantRows = convertTriqlerInputToPeptQuantRows(triqlerInputFile, peptQuantRowFile, params)
  if params['t-test']:
    qvalMethod = 'pvalues'
  else:
    qvalMethod = 'avg_pep'
    
  selectComparisonBayesTmp = lambda proteinOutputRows, comparisonKey : selectComparisonBayes(proteinOutputRows, comparisonKey, params['t-test'])
  diff_exp.doDiffExp(params, peptQuantRows, triqlerOutputFile, getPickedProteinCalibration, selectComparisonBayesTmp, qvalMethod = qvalMethod)

def convertTriqlerInputToPeptQuantRows(triqlerInputFile, peptQuantRowFile, params):
  peptQuantRowMap, getPEPFromScore, params['fileList'], params['groupLabels'], params['groups'] = getPeptQuantRowMap(triqlerInputFile, params['decoyPattern'])
  
  spectrumToFeatureMatch, featureClusterRows, intensityDiv = selectBestFeaturesPerRunAndSpectrum(peptQuantRowMap, getPEPFromScore, params)
  
  peptideQuantRows = selectBestPqrPerFeatureCluster(spectrumToFeatureMatch, featureClusterRows, intensityDiv)
  
  peptideQuantRows = updateIdentPEPs(peptideQuantRows, params['decoyPattern'])
  
  print("Writing peptide quant rows to file")
  printPeptideQuantRows(peptQuantRowFile, parsers.getRunIds(params), peptideQuantRows)
  
  return peptideQuantRows

def getPeptQuantRowMap(triqlerInputFile, decoyPattern):
  peptQuantRowMap = collections.defaultdict(list)
  seenSpectra = set()
  targetScores, decoyScores = list(), list()
  runCondPairs = list()
  for trqRow in parsers.parseTriqlerInputFile(triqlerInputFile):
    peptQuantRowMap[trqRow.featureClusterId].append(trqRow)
    if (trqRow.run, trqRow.condition) not in runCondPairs:
      runCondPairs.append((trqRow.run, trqRow.condition))
    
    if trqRow.spectrumId not in seenSpectra:
      if isDecoy(trqRow.proteins, decoyPattern):
        decoyScores.append(trqRow.searchScore)
      else:
        targetScores.append(trqRow.searchScore)
      seenSpectra.add(trqRow.spectrumId)
  
  if len(decoyScores) == 0:
    sys.exit("ERROR: No decoy hits found, check if the correct decoy prefix was specified with the --decoy_pattern flag")
  
  targetScores = sorted(targetScores, reverse = True)
  decoyScores = sorted(decoyScores, reverse = True)
  _, peps = qvality.getQvaluesFromScores(targetScores, decoyScores, includePEPs = True, includeDecoys = True, tdcInput = True)
  peps = peps[::-1]
  
  allScores = np.array(sorted(targetScores + decoyScores))
  getPEPFromScore = lambda score : peps[min(np.searchsorted(allScores, score, side = 'left'), len(peps) - 1)]
  
  fileList, groupLabels, groups = getFilesAndGroups(runCondPairs)
  
  if len(groups) < 2:
    sys.exit("ERROR: At least two treatment groups should be specified")
    
  return peptQuantRowMap, getPEPFromScore, fileList, groupLabels, groups

def selectBestFeaturesPerRunAndSpectrum(peptQuantRowMap, getPEPFromScore, params):
  minIntensity = 1e100
  featureClusterRows = list()
  spectrumToFeatureMatch = dict() # stores the best peptideQuantRow per (peptide, spectrumIdx)-pair
  for featureClusterIdx, trqRows in peptQuantRowMap.items():
    if featureClusterIdx % 10000 == 0:
      print("featureClusterIdx:", featureClusterIdx)

    bestPEP = collections.defaultdict(lambda : [(1.01, None)]*len(params['fileList'])) # peptide => array([linkPEP, precursorCandidate])
    bestPeptideScore = collections.defaultdict(lambda : (-1e9, -1)) # peptide => (identPEP, spectrumIdx); contains the identification PEPs
    
    for trqRow in trqRows:
      fileIdx = params['fileList'].index(trqRow.run)
      if trqRow.peptide != "NA" and (trqRow.linkPEP < bestPEP[trqRow.peptide][fileIdx][0] or (trqRow.linkPEP == bestPEP[trqRow.peptide][fileIdx][0] and trqRow.intensity > bestPEP[trqRow.peptide][fileIdx][1].intensity)):
        bestPEP[trqRow.peptide][fileIdx] = (trqRow.linkPEP, trqRow)
        bestPeptideScore[trqRow.peptide] = max([bestPeptideScore[trqRow.peptide], (trqRow.searchScore, trqRow.spectrumId)])
    
    for peptide in bestPEP:
      if sum(1 for x in bestPEP[peptide] if x[0] < 1.0) < params['minSamples']:
        continue
      intensities = [0.0]*len(params['fileList'])
      linkPEPs = [1.01]*len(params['fileList'])
      peptLinkEP = 1.0
      first = True
      #svmScore, spectrumId = bestPeptideScore[peptide]
      #identPEP = getPEPFromScore(svmScore)
      for fileIdx, (linkPEP, trqRow) in enumerate(bestPEP[peptide]):
        if trqRow != None:
          if first:
            charge, proteins, first = trqRow.charge, trqRow.proteins, False
          identPEP = getPEPFromScore(trqRow.searchScore)
          peptLinkEP *= 1.0 - linkPEP
          linkPEPs[fileIdx] = combinePEPs(linkPEP, identPEP)
          intensities[fileIdx] = trqRow.intensity
      
      # set the linkPEP for missing values to the max linkPEP in the row
      maxLinkPEP = max([x for x in linkPEPs if x <= 1.0])
      linkPEPs = [x if x <= 1.0 else maxLinkPEP for x in linkPEPs]
      
      minIntensity = min(minIntensity, min([x for x in intensities if x > 0.0]))
      
      #linkPEPs = [combinedPEP]*len(params['fileList']) # use peptide-linkPEP instead of run-linkPEP
      #svmScore = -1*np.log(combinedPEP) # take into account run-linkPEPs for identification PEP
      #svmScore = np.log(1.0 / identPEP - 1.0) # use PEP score
      
      svmScore, spectrumId = bestPeptideScore[peptide]
      featureClusterRows.append((intensities, featureClusterIdx, spectrumId, linkPEPs, peptide, proteins, svmScore, charge))
      
      # multiple featureClusters can be associated with the same consensus spectrum
      # when two or more analytes match closely in prec m/z and retention time;
      # choose the best featureCluster per (peptide, spectrum)-pair based on combinedPEP
      # note that chimeric spectra can still be associated with multiple peptideQuantRows, 
      # as the peptide is included in the key
      identPEP = getPEPFromScore(svmScore)
      combinedPEP = combinePEPs(identPEP, 1.0 - peptLinkEP)
      key = (parsers.cleanPeptide(peptide), spectrumId / 100)
      if combinedPEP < spectrumToFeatureMatch.get(key, (-1, -1, 1.01))[2]:
        spectrumToFeatureMatch[key] = (spectrumId, featureClusterIdx, combinedPEP)
  
  # divide intensities by a power of 10 for increased readability of peptide 
  # output file, make sure that the lowest intensity retains two significant 
  # digits after printing with two digits after the decimal point
  intensityDiv = np.power(10, np.floor(np.log10(minIntensity))+1)
  print("Dividing intensities by %g for increased readability" % intensityDiv)
  
  return spectrumToFeatureMatch, featureClusterRows, intensityDiv
  
def getFilesAndGroups(runCondPairs):
  runCondPairs = sorted(runCondPairs, key = lambda x : (int(x[1].split(":")[0]), x[0]))
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
def selectBestPqrPerFeatureCluster(spectrumToFeatureMatch, featureClusterRows, intensityDiv = 1e6):
  featureClusterToSpectrumIdx = dict()
  for (spectrumIdx, featureClusterIdx, combinedPEP) in spectrumToFeatureMatch.values():
    if combinedPEP < featureClusterToSpectrumIdx.get(featureClusterIdx, (-1, 1.01))[1]:
      featureClusterToSpectrumIdx[featureClusterIdx] = (spectrumIdx, combinedPEP)
  survivingSpectrumIdxs = set([y[0] for x, y in featureClusterToSpectrumIdx.items()])
  print("Surviving spectrumIdxs:", len(survivingSpectrumIdxs))
  
  print("Converting to peptide quant rows")
  peptideQuantRows = list()
  quantifiableProteins = list()
  for intensities, featureClusterIdx, spectrumIdx, linkPEPs, peptide, proteins, svmScore, charge in featureClusterRows:
    if spectrumIdx in survivingSpectrumIdxs:
      row = parsers.PeptideQuantRow(0.0, charge, featureClusterIdx, linkPEPs, list(map(lambda x : x/intensityDiv, intensities)), svmScore, peptide, proteins)
      peptideQuantRows.append(row)
  return peptideQuantRows
  
def printPeptideQuantRows(peptOutputFile, headers, peptideQuantRows):
  writer = csv.writer(open(peptOutputFile, 'w'), delimiter = '\t')
  writer.writerow(parsers.getPeptideQuantRowHeaders(headers))
  for row in peptideQuantRows:
    writer.writerow(row.toList())
     
def getProteinCalibration(peptQuantRows, proteinModifier, getEvalFeatures, decoyPattern):
  protQuantRows = parsers.filterAndGroupPeptides(peptQuantRows)
  
  proteinTargetRows, proteinDecoyRows = list(), list()
  for prot, quantRows in protQuantRows:
    psmIdx = 0
    quantRows = sorted(quantRows, key = lambda x : x.combinedPEP)
    seenPeptides = set()
    usablePeptides = 0
    filteredQuantRows = list()
    for quantRow in quantRows:
      if quantRow.peptide not in seenPeptides:
        seenPeptides.add(quantRow.peptide)
        usablePeptides += 1
        filteredQuantRows.append(quantRow)

    if usablePeptides < 1:
      continue
    else:
      quantRows = filteredQuantRows

    isDecoyProt = isDecoy([prot], decoyPattern)
    protein = proteinModifier(prot)
    #evalFeatures = getEvalFeatures(quantRows[psmIdx].quant)
    #numPeptides = len(quantRows)
    evalFeatures = list()
    numPeptides = usablePeptides

    proteinOutputRow = (quantRows[psmIdx].linkPEP, protein, quantRows, evalFeatures, numPeptides)
    if isDecoyProt:
      proteinDecoyRows.append(proteinOutputRow)
    else:
      proteinTargetRows.append(proteinOutputRow)

  proteinTargetRows.sort()
  proteinDecoyRows.sort()
  return proteinTargetRows, proteinDecoyRows

def getPickedProteinCalibration(peptQuantRows, params, proteinModifier, getEvalFeatures):
  targetProteinOutputRows, decoyProteinOutputRows = getProteinCalibration(peptQuantRows, proteinModifier, getEvalFeatures, params['decoyPattern'])

  pickedProteinOutputRows = targetProteinOutputRows + decoyProteinOutputRows
  np.random.shuffle(pickedProteinOutputRows)

  proteinQuantIdPEP = lambda quantRows : np.sum([np.log(x.identificationPEP) for x in quantRows])
  pickedProteinOutputRows = sorted(pickedProteinOutputRows, key = lambda x : proteinQuantIdPEP(x[2]))
  
  print("Fitting hyperparameters")
  hyperparameters.fitPriors(peptQuantRows, params) # updates priors
  
  targetScores, decoyScores = list(), list()
  proteinOutputRows = list()
  seenProteins = set()
  
  print("Calculating protein quants")
  processingPool = pool.MyPool(params['numThreads'])
  pickedProteinOutputRowsNew = list()
  for linkPEP, protein, quantRows, evalFeatures, numPeptides in pickedProteinOutputRows:
    evalProtein = protein.replace(params['decoyPattern'], "", 1)
    if evalProtein not in seenProteins:
      seenProteins.add(evalProtein)
      
      #score = calibration.scoreFromPEP(proteinQuantIdPEP(quantRows))
      score = -1*proteinQuantIdPEP(quantRows)
      if isDecoy([protein], params['decoyPattern']):
        decoyScores.append(score)
      else:
        targetScores.append(score)
      pickedProteinOutputRowsNew.append([linkPEP, protein, quantRows, evalFeatures, numPeptides])
      processingPool.applyAsync(pgm.getPosteriors, [quantRows, params])
      #pgm.getPosteriors(quantRows, params) # for debug mode
  posteriors = processingPool.checkPool(printProgressEvery = 50)
  
  _, peps = qvality.getQvaluesFromScores(targetScores, decoyScores, includePEPs = True, includeDecoys = True, tdcInput = True)
  
  proteinOutputRowsUpdatedPEP = list()
  sumPEP = 0.0
  for (linkPEP, protein, quantRows, evalFeatures, numPeptides), (bayesQuantRow, muGroupDiffs, probsBelowFoldChange), proteinPEP in zip(pickedProteinOutputRowsNew, posteriors, peps):
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
def updateIdentPEPs(peptideQuantRow, decoyPattern):
  scoreIdxPairs = list()
  for i, row in enumerate(peptideQuantRow):
    scoreIdxPairs.append([row.identificationPEP, i, isDecoy(row.protein, decoyPattern)]) # row.identificationPEP contains the SVM score
  
  scoreIdxPairs = sorted(scoreIdxPairs, reverse = True)
  scoreIdxs = np.argsort([x[1] for x in scoreIdxPairs])
  targetScores = [x[0] for x in scoreIdxPairs if x[2] == False]
  decoyScores = [x[0] for x in scoreIdxPairs if x[2] == True]
  
  _, identPEPs = qvality.getQvaluesFromScores(targetScores, decoyScores, includePEPs = True, includeDecoys = True, tdcInput = True)
  
  newPeptideQuantRows = list()
  for i, row in enumerate(peptideQuantRow):
    identPEP = identPEPs[scoreIdxs[i]]
    newPeptideQuantRows.append(row._replace(identificationPEP = identPEP, combinedPEP = identPEP, linkPEP = [combinePEPs(identPEP, x) for x in row.linkPEP])) # using single spectra
    #newPeptideQuantRows.append(row._replace(identificationPEP = identPEP, combinedPEP = identPEP)) # using consensus spectra
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
