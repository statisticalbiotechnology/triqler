from __future__ import print_function

"""triqler.triqler: provides entry point main()."""

__version__ = "1.0.0"

import sys
import random
import collections
import getopt
import itertools
import copy
import re
import csv

import numpy as np

from . import parsers
from . import percolator
from . import qvality
from . import hyperparameters
from . import multiprocessing_pool as pool
from . import pgm
from . import diff_exp

def main(argv):
  helpMessage = 'triqler.py -i <triqler_input_file> -o <triqler_output_file> -e <min_fold_change>'

  try:
    opts, args = getopt.getopt(argv,"hi:e:",["ifile=","minfoldchange="])
  except getopt.GetoptError:
    print(helpMessage)
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print(helpMessage)
      sys.exit()
    elif opt in ("-i", "--ifile"):
      triqlerInputFile = arg
    elif opt in ("-e", "--minfoldchange"):
      foldChangeEval = float(arg)
  
  params = dict()
  params['foldChangeEval'] = foldChangeEval
  params['t-test'] = False
  
  runTriqler(params, triqlerInputFile)

def runTriqler(params, triqlerInputFile):  
  peptQuantRowFile = triqlerInputFile + ".pqr.tsv"
  peptQuantRows = convertTriqlerInputToPeptQuantRows(triqlerInputFile, peptQuantRowFile, params)
  if params['t-test']:
    qvalMethod = 'pvalues'
  else:
    qvalMethod = 'avg_pep'
    
  selectComparisonBayesTmp = lambda proteinOutputRows, comparisonKey : selectComparisonBayes(proteinOutputRows, comparisonKey, params['t-test'])
  diff_exp.doDiffExp(params, peptQuantRows, triqlerInputFile, getPickedProteinCalibration, selectComparisonBayesTmp, qvalMethod = qvalMethod)

def convertTriqlerInputToPeptQuantRows(triqlerInputFile, peptQuantRowFile, params):
  peptQuantRowMap = collections.defaultdict(list)
  
  seenSpectra = set()
  targetScores, decoyScores = list(), list()
  runCondPairs = list()
  for trqRow in parsers.parseTriqlerInputFile(triqlerInputFile):
    peptQuantRowMap[trqRow.featureClusterId].append(trqRow)
    if (trqRow.run, trqRow.condition) not in runCondPairs:
      runCondPairs.append((trqRow.run, trqRow.condition))
    
    if trqRow.spectrumId not in seenSpectra:
      if isDecoy(trqRow.proteins):
        decoyScores.append(trqRow.searchScore)
      else:
        targetScores.append(trqRow.searchScore)
      seenSpectra.add(trqRow.spectrumId)
  
  runCondPairs = sorted(runCondPairs, key = lambda x : (int(x[1].split(":")[0]), x[0]))
  fileList = list()
  params['groupLabels'], params['groups'] = list(), list()
  for run, cond in runCondPairs:
    if run not in fileList:
      fileList.append(run)
      if cond not in params['groupLabels']:
        params['groupLabels'].append(cond)
        params['groups'].append([])
      params['groups'][params['groupLabels'].index(cond)].append(len(fileList) - 1)
    
    
  targetScores = sorted(targetScores, reverse = True)
  decoyScores = sorted(decoyScores, reverse = True)
  _, peps = percolator.getQvaluesFromScores(targetScores, decoyScores, includePEPs = True, includeDecoys = True, tdcInput = True)
  #peps = qvality.getQvaluesFromScores(targetScores, decoyScores)
  peps = peps[::-1]
  
  allScores = np.array(sorted(targetScores + decoyScores))
    
  minIntensity = 1e100
  featureClusterRows = list()
  spectrumToFeatureMatch = dict() # stores the best peptideQuantRow per (peptide, spectrumIdx)-pair
  for featureClusterIdx, trqRows in peptQuantRowMap.items():
    if featureClusterIdx % 10000 == 0:
      print("featureClusterIdx:", featureClusterIdx)

    bestPEP = collections.defaultdict(lambda : [(1.01, None)]*len(fileList)) # peptide => array([linkPEP, precursorCandidate])
    bestPeptideScore = collections.defaultdict(lambda : (-1e9, -1)) # peptide => (identPEP, spectrumIdx); contains the identification PEPs
    
    for trqRow in trqRows:
      fileIdx = fileList.index(trqRow.run)
      if trqRow.peptide != "NA" and (trqRow.linkPEP < bestPEP[trqRow.peptide][fileIdx][0] or (trqRow.linkPEP == bestPEP[trqRow.peptide][fileIdx][0] and trqRow.intensity > bestPEP[trqRow.peptide][fileIdx][1].intensity)):
        bestPEP[trqRow.peptide][fileIdx] = (trqRow.linkPEP, trqRow)
        bestPeptideScore[trqRow.peptide] = max([bestPeptideScore[trqRow.peptide], (trqRow.searchScore, trqRow.spectrumId)])
      
    for peptide in bestPEP:
      intensities = [0.0]*len(fileList)
      expMasses = list()
      linkPEPs = [1.01]*len(fileList)
      peptLinkEP = 1.0
      first = True
      #svmScore, spectrumId = bestPeptideScore[peptide]
      #identPEP = peps[min(np.searchsorted(allScores, svmScore, side = 'left'), len(peps) - 1)]
      for linkPEP, trqRow in bestPEP[peptide]:
        if trqRow != None:
          if first:
            #charge, rTime, proteins, first = trqRow.charge, trqRow.rTime, trqRow.proteins, False
            charge, rTime, proteins, first = trqRow.charge, 0.0, trqRow.proteins, False
          identPEP = peps[min(np.searchsorted(allScores, trqRow.searchScore, side = 'left'), len(peps) - 1)]
          peptLinkEP *= 1.0 - linkPEP
          fileIdx = fileList.index(trqRow.run)
          linkPEPs[fileIdx] = combinePEPs(linkPEP, identPEP)
          intensities[fileIdx] = trqRow.intensity
          #expMasses.append(trqRow.precMz)
          expMasses.append(0.0)
      
      maxLinkPEP = max([x for x in linkPEPs if x <= 1.0])
      linkPEPs = [x if x <= 1.0 else maxLinkPEP for x in linkPEPs]
      combinedPEP = 1.0 - peptLinkEP
      expMass = np.mean(expMasses)
      
      minIntensity = min(minIntensity, min([x for x in intensities if x > 0.0]))
      
      #linkPEPs = [combinedPEP]*len(fileList) # use peptide-linkPEP instead of run-linkPEP
      #svmScore = -1*np.log(combinedPEP) # take into account run-linkPEPs for identification PEP
      #svmScore = np.log(1.0 / identPEP - 1.0) # use PEP score
      
      svmScore, spectrumId = bestPeptideScore[peptide]
      featureClusterRows.append((intensities, featureClusterIdx, spectrumId, linkPEPs, peptide, proteins, svmScore, expMass, charge, rTime))
      
      # multiple featureClusters can be associated with the same consensus spectrum
      # when two or more analytes match closely in prec m/z and retention time;
      # choose the best featureCluster per (peptide, spectrum)-pair based on combinedPEP
      # note that chimeric spectra can still be associated with multiple peptideQuantRows, 
      # as the peptide is included in the key
      identPEP = peps[min(np.searchsorted(allScores, svmScore, side = 'left'), len(peps) - 1)]
      combinedPEP = combinePEPs(identPEP, combinedPEP)
      key = (cleanPeptide(peptide), spectrumId / 100)
      if combinedPEP < spectrumToFeatureMatch.get(key, (-1, -1, 1.01))[2]:
        spectrumToFeatureMatch[key] = (spectrumId, featureClusterIdx, combinedPEP)
        
  # multiple peptideQuantRows can be associated with the same featureCluster 
  # when two or more analytes match closely in prec m/z and retention time;
  # choose the best peptideQuantRow per featureCluster based on combinedPEP
  featureClusterToSpectrumIdx = dict()
  for (spectrumIdx, featureClusterIdx, combinedPEP) in spectrumToFeatureMatch.values():
    if combinedPEP < featureClusterToSpectrumIdx.get(featureClusterIdx, (-1, 1.01))[1]:
      featureClusterToSpectrumIdx[featureClusterIdx] = (spectrumIdx, combinedPEP)
  survivingSpectrumIdxs = set([y[0] for x, y in featureClusterToSpectrumIdx.items()])
  print("Surviving spectrumIdxs:", len(survivingSpectrumIdxs))
  
  # divide intensities by a power of 10 for increased readability of peptide 
  # output file, make sure that the lowest intensity retains two significant 
  # digits after printing with two digits after the decimal point
  intensityDiv = np.power(10, np.floor(np.log10(minIntensity))+1)
  #intensityDiv = 1e6
  print("Dividing intensities by %g for increased readability" % intensityDiv)
  
  print("Converting to peptide quant rows")
  peptideQuantRows = list()
  quantifiableProteins = list()
  for intensities, featureClusterIdx, spectrumIdx, linkPEPs, peptide, proteins, svmScore, expMass, charge, rTime in featureClusterRows:
    if spectrumIdx in survivingSpectrumIdxs:
      row = parsers.PeptideQuantRow(0.0, linkPEPs, expMass, charge, rTime, featureClusterIdx, list(map(lambda x : x/intensityDiv, intensities)), ",".join(proteins), svmScore, peptide)
      peptideQuantRows.append(row)
  
  peptideQuantRows = updateIdentPEPs(peptideQuantRows)
  
  print("Writing peptide quant rows to file")
  printPeptideQuantRows(peptQuantRowFile, [parsers.getGroupLabel(idx, params['groups'], params['groupLabels']) + ":" + x.split("/")[-1] for idx, x in enumerate(fileList)], peptideQuantRows)
  
  return peptideQuantRows

def printPeptideQuantRows(peptOutputFile, headers, peptideQuantRows):
  writer = csv.writer(open(peptOutputFile, 'w'), delimiter = '\t')
  writer.writerow(parsers.getPeptideQuantRowHeaders(headers))
  for row in peptideQuantRows:
    writer.writerow(row.toList())
     
def getProteinCalibration(peptQuantRows, proteinModifier, getEvalFeatures):
  peptQuantRows = filter(lambda x : x.protein != "NA" and x.protein.count(",") == 0 and x.combinedPEP < 1.0, peptQuantRows)

  protQuantRows = itertools.groupby(sorted(peptQuantRows, key = lambda x : x.protein), key = lambda x : x.protein)
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

    isDecoy = True
    for protTmp in prot.split(','):
      if not "decoy_" in protTmp:
        isDecoy = False
        break
    protein = proteinModifier(prot)
    #evalFeatures = getEvalFeatures(quantRows[psmIdx].quant)
    #numPeptides = len(quantRows)
    evalFeatures = list()
    numPeptides = usablePeptides

    proteinOutputRow = (quantRows[psmIdx].linkPEP, protein, quantRows, evalFeatures, numPeptides)
    if isDecoy:
      proteinDecoyRows.append(proteinOutputRow)
    else:
      proteinTargetRows.append(proteinOutputRow)

  proteinTargetRows.sort()
  proteinDecoyRows.sort()
  return proteinTargetRows, proteinDecoyRows

def getPickedProteinCalibration(peptQuantRows, params, proteinModifier, getEvalFeatures):
  targetProteinOutputRows, decoyProteinOutputRows = getProteinCalibration(peptQuantRows, proteinModifier, getEvalFeatures)

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
  processingPool = pool.MyPool(4)
  pickedProteinOutputRowsNew = list()
  for linkPEP, protein, quantRows, evalFeatures, numPeptides in pickedProteinOutputRows:
    evalProtein = protein.replace("decoy_", "")
    if evalProtein not in seenProteins:
      seenProteins.add(evalProtein)
      
      #score = calibration.scoreFromPEP(proteinQuantIdPEP(quantRows))
      score = -1*proteinQuantIdPEP(quantRows)
      if "decoy_" in protein:
        decoyScores.append(score)
      else:
        targetScores.append(score)
      pickedProteinOutputRowsNew.append([linkPEP, protein, quantRows, evalFeatures, numPeptides])
      processingPool.applyAsync(pgm.getPosteriors, [quantRows, params])
  posteriors = processingPool.checkPool(printProgressEvery = 50)
  
  _, peps = percolator.getQvaluesFromScores(targetScores, decoyScores, includePEPs = True, includeDecoys = True, tdcInput = True)
  #peps = qvality.getQvaluesFromScores(targetScores, decoyScores)
  
  proteinOutputRowsUpdatedPEP = list()
  sumPEP = 0.0
  for (linkPEP, protein, quantRows, evalFeatures, numPeptides), (bayesQuantRow, mus, sigmas, probsBelowFoldChange), proteinPEP in zip(pickedProteinOutputRowsNew, posteriors, peps):
    evalFeatures = getEvalFeatures(bayesQuantRow)
    if not params['t-test']:
      evalFeatures[-1] = probsBelowFoldChange
    
    if not params['t-test'] or sumPEP / (len(proteinOutputRowsUpdatedPEP) + 1) < 0.05:
      proteinOutputRowsUpdatedPEP.append([linkPEP, protein, quantRows, evalFeatures, numPeptides, proteinPEP])
      sumPEP += proteinPEP
  
  proteinOutputRowsUpdatedPEP = sorted(proteinOutputRowsUpdatedPEP, key = lambda x : (x[0], x[1]))
  return proteinOutputRowsUpdatedPEP

def selectComparisonBayes(proteinOutputRows, comparisonKey, tTest = False):
  proteinOutputRowsUpdatedPEP = list()
  for (linkPEP, protein, quantRows, evalFeatures, numPeptides, proteinPEP) in proteinOutputRows:
    evalFeaturesNew = copy.deepcopy(evalFeatures)
    evalFeaturesNew[-1] = evalFeatures[-1][comparisonKey] # probBelowFoldChange
    evalFeaturesNew[-2] = evalFeatures[-2][comparisonKey] # log2_fold_change
    if not tTest:
      combinedPEP = combinePEPs(evalFeaturesNew[-1], proteinPEP)
    else:
      combinedPEP = evalFeaturesNew[-1]
    
    proteinOutputRowsUpdatedPEP.append([combinedPEP, linkPEP, protein, quantRows, evalFeaturesNew, numPeptides, proteinPEP])

  proteinOutputRowsUpdatedPEP = sorted(proteinOutputRowsUpdatedPEP, key = lambda x : (x[0], x[1]))
  return proteinOutputRowsUpdatedPEP

def updateIdentPEPs(peptideQuantRow):
  scoreIdxPairs = list()
  for i, row in enumerate(peptideQuantRow):
    scoreIdxPairs.append([row.identificationPEP, i, isDecoy(row.protein.split(","))]) # row.identificationPEP contains the SVM score
  
  scoreIdxPairs = sorted(scoreIdxPairs, reverse = True)
  scoreIdxs = np.argsort([x[1] for x in scoreIdxPairs])
  targetScores = [x[0] for x in scoreIdxPairs if x[2] == False]
  decoyScores = [x[0] for x in scoreIdxPairs if x[2] == True]
  
  _, identPEPs = percolator.getQvaluesFromScores(targetScores, decoyScores, includePEPs = True, includeDecoys = True, tdcInput = True)
  #identPEPs = qvality.getQvaluesFromScores(targetScores, decoyScores)
  
  newPeptideQuantRows = list()
  for i, row in enumerate(peptideQuantRow):
    identPEP = identPEPs[scoreIdxs[i]]
    newPeptideQuantRows.append(row._replace(identificationPEP = identPEP, combinedPEP = identPEP, linkPEP = [combinePEPs(identPEP, x) for x in row.linkPEP])) # using single spectra
    #newPeptideQuantRows.append(row._replace(identificationPEP = identPEP, combinedPEP = identPEP)) # using consensus spectra
  return newPeptideQuantRows
    
def cleanPeptide(peptide):
  return re.sub('\[[-0-9]*\]', '', peptide[2:-2])
 
def isDecoy(proteins):
  isDecoy = True
  for protein in proteins:
    if not "decoy_" in protein:
      isDecoy = False
      break
  return isDecoy
  
def combinePEPs(linkPEP, identPEP):
  return 1.0 - (1.0 - linkPEP)*(1.0 - identPEP)

def scoreFromPEP(PEP):
  return -1*np.log(PEP + 1e-18)

if __name__ == "__main__":
  main(sys.argv[1:])
