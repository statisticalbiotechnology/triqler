#!/usr/bin/python

from __future__ import print_function

import itertools

import numpy as np
from scipy.stats import f_oneway, kruskal

from . import parsers
from . import qvality

def doDiffExp(params, peptQuantRows, outputFile, proteinQuantificationMethod, selectComparison, qvalMethod):    
  proteinModifier, getEvalFeatures, evalFunctions = getEvalFunctions(outputFile, params)
  
  proteinOutputRows = proteinQuantificationMethod(peptQuantRows, params, proteinModifier, getEvalFeatures)
  
  if len(params['proteinPosteriorsOutput']) > 0:
    printProteinPosteriors(proteinOutputRows, params)
    
  if len(params['groupPosteriorsOutput']) > 0:
    printGroupPosteriors(proteinOutputRows, params)
  
  if len(params['foldChangePosteriorsOutput']) > 0:
    printFoldChangePosteriors(proteinOutputRows, params)
  
  numGroups = len(params['groups'])
  for groupId1, groupId2 in itertools.combinations(range(numGroups), 2):
    if numGroups == 2:
      proteinOutputFile = outputFile
    else:
      proteinOutputFile = getOutputFile(outputFile, groupId1, groupId2)
    
    print("Comparing", params['groupLabels'][groupId1], "to", params['groupLabels'][groupId2])
    print("  output file:", proteinOutputFile)
    
    proteinOutputRowsGroup = selectComparison(proteinOutputRows, (groupId1, groupId2))
    if "trueConcentrationsDict" in params and len(params["trueConcentrationsDict"]) > 0:
      evalFunctions = [lambda protein, evalFeatures : evalTruePositiveTtest(params["trueConcentrationsDict"], protein, groupId1, groupId2, evalFeatures[-2], params)]
    
    printProteinQuantRows(proteinOutputRowsGroup, qvalMethod, evalFunctions, proteinOutputFile, params)

def getOutputFileExtension(outputFile):
  fileName = outputFile.split("/")[-1]
  if "." in fileName:
    return outputFile, "." + fileName.split(".")[-1]
  else:
    return outputFile + ".", "."

def getOutputFile(outputFile, groupId1, groupId2):
  outputFile, outputFileExt = getOutputFileExtension(outputFile)
  return outputFile.replace(outputFileExt, ".%dvs%d%s" % (groupId1 + 1, 
      groupId2 + 1, outputFileExt if len(outputFileExt) > 1 else ""))
  
def getTrueConcentrations(trueConcentrationsDict, protein):
  for key, value in trueConcentrationsDict.items():
    if key in protein:
      return value
  return []
      
def evalTruePositiveANOVA(trueConcentrationsDict, protein):
  trueConcentrations = getTrueConcentrations(trueConcentrationsDict, protein)
  return len(trueConcentrations) > 0

def evalTruePositiveTtest(trueConcentrationsDict, protein, groupId1, groupId2, foldChange, params):
  trueConcentrations = getTrueConcentrations(trueConcentrationsDict, protein)
  if len(trueConcentrations) > 0:
    trueRatios = np.dot(np.matrix(trueConcentrations).T, np.ones_like(np.matrix(trueConcentrations)))
    trueLogRatios = np.log2(trueRatios / trueRatios.T)
    return np.abs(trueLogRatios[groupId1, groupId2]) > params['foldChangeEval'] and \
             trueLogRatios[groupId1, groupId2]*foldChange > 0
  else:
    return False
      
def getEvalFunctions(peptInputFile, params):
  getDEEvalFeatures = lambda quant : [getFoldChange(quant, params), getDiffExp(quant, params)]
  getEvalFeatures = getDEEvalFeatures
  proteinModifier = lambda protein : protein
  
  evalFunctions = []
  return proteinModifier, getEvalFeatures, evalFunctions

def getDiffExp(quants, params):
  quants = parsers.getQuantGroups(quants, params['groups'])
  quantsNotNan = list()
  for q in quants:
    quantsNotNan.append([x for x in q if not np.isnan(x) and not np.isinf(x)])
  
  anovaPvalues = dict()
  anovaPvalues['ANOVA'] = getPval(quantsNotNan)
  
  numGroups = len(params['groups'])
  if numGroups > 2:
    for groupId1, groupId2 in itertools.combinations(range(numGroups), 2):
      quantsNotNanFiltered = [quantsNotNan[groupId1], quants[groupId2]]
      anovaPvalues[(groupId1, groupId2)] = getPval(quantsNotNanFiltered)
  return anovaPvalues
  
def getPval(quants):
  anovaFvalue, anovaPvalue = f_oneway(*quants)
  if not np.isnan(anovaPvalue):
    return anovaPvalue
  else:
    #print(quants)
    return 1.0
    
def getFoldChange(quants, params):
  foldChange = dict()
  
  numGroups = len(params['groups'])
  if numGroups > 2:
    maxFoldChange = 0.0
    for groupId1, groupId2 in itertools.combinations(range(numGroups), 2):
      foldChange[(groupId1, groupId2)] = getFc(quants, params, groupId1, groupId2)
      maxFoldChange = np.max([maxFoldChange, np.abs(foldChange[(groupId1, groupId2)])])
    foldChange['ANOVA'] = maxFoldChange
  elif numGroups == 2:
    foldChange['ANOVA'] = getFc(quants, params, 0, 1)
  return foldChange

def getFc(quants, params, groupId1, groupId2):
  return np.log2(np.mean([quants[x] for x in params['groups'][groupId1]]) / np.mean([quants[x] for x in params['groups'][groupId2]]))
  
def printProteinQuantRows(proteinOutputRows, qvalMethod, evalFunctions, outputFile, params, qvalThreshold = 0.05):
  writer = parsers.getTsvWriter(outputFile)
  
  evalHeaders = ["log2_fold_change", "diff_exp_prob_" + str(params['foldChangeEval'])]
  if 'pvalues' in qvalMethod:
    evalHeaders[1] = "diff_exp_pval_" + str(params['foldChangeEval'])
    targetPvalues = [x[4] for x in proteinOutputRows]
    reportedQvalsPval, reportedPEPsPval = qvality.getQvaluesFromPvalues(targetPvalues, includePEPs = True)
  
  protOutputHeaders = ["posterior_error_prob", "protein", "num_peptides", "protein_id_posterior_error_prob"] + evalHeaders + parsers.getRunIds(params) + ["peptides"]
  checkCalibration = len(evalFunctions) > 0
  if checkCalibration:
    evalTruePositives = evalFunctions[0]
    writer.writerow(["observed_q_value", "reported_q_value"] + protOutputHeaders)
  else:
    writer.writerow(["q_value"] + protOutputHeaders)
  
  outRows = list()
  observedQvals, reportedQvals, reportedPEPs = list(), list(), list()
  qval, observedQval, sumPEP = 0.0, 0.0, 0.0
  targets, fp, tp = 0, 1, 0
  numTies = 1
  nextScores = [x[0] for x in proteinOutputRows] + [np.nan]
  numSignificant = 0
  for i, (combinedPEP, _, protein, quantRows, evalFeatures, numPeptides, proteinIdPEP, quants, _) in enumerate(proteinOutputRows):
    if 'pvalues_with_fc' in qvalMethod and np.abs(evalFeatures[-2]) < params['foldChangeEval']:
      continue

    if not protein.startswith(params['decoyPattern']):
      sumPEP += combinedPEP
      targets += 1
      qval = sumPEP / targets
      if qval < qvalThreshold:
        numSignificant += 1
      
      if checkCalibration:
        if evalTruePositives(protein, evalFeatures):
          tp += 1
        else:
          fp += 1
        observedQval = float(fp) / (tp+fp)

    if combinedPEP == nextScores[i+1]:
      numTies += 1
    else:
      for _ in range(numTies):
        if 'pvalues' in qvalMethod:
          reportedQvals.append(reportedQvalsPval[i])
        else:
          reportedQvals.append(qval)
        
        if checkCalibration:
          observedQvals.append(observedQval)
      numTies = 1
    
    if 'pvalues' in qvalMethod:
      combinedPEP = reportedPEPsPval[i]
    
    outRows.append(["%.4g" % combinedPEP, protein, numPeptides, "%.4g" % proteinIdPEP] + ["%.4g" % x for x in evalFeatures] + ["%.4g" % x for x in quants] + [x.peptide for x in quantRows])
  
  if checkCalibration:
    observedQvals = qvality.fdrsToQvals(observedQvals)
    for outRow, observedQval, reportedQval in zip(outRows, observedQvals, reportedQvals):
      writer.writerow(["%.4g" % (observedQval), "%.4g" % (reportedQval)] + outRow)
  else:
    for outRow, reportedQval in zip(outRows, reportedQvals):
      writer.writerow(["%.4g" % (reportedQval)] + outRow)
  
  print("  Found", numSignificant, "target proteins as differentially abundant at", str(int(qvalThreshold * 100)) + "% FDR")

def printProteinPosteriors(proteinOutputRows, params):
  print("Writing protein posteriors to", params['proteinPosteriorsOutput'])
  writer = parsers.getTsvWriter(params['proteinPosteriorsOutput'])
  
  writer.writerow(["protein", "group:run"] + ['%.4g' % x for x in params['proteinQuantCandidates']])
  for i, (_, protein, _, _, _, _, _, posteriorDists) in enumerate(proteinOutputRows):
    if posteriorDists:
      pProteinQuantsList, _, _ = posteriorDists
      for run, posterior in zip(parsers.getRunIds(params), pProteinQuantsList):
        writer.writerow([protein, run] + ['%.4g' % p for p in posterior])
    
def printGroupPosteriors(proteinOutputRows, params):
  print("Writing treatment group posteriors to", params['groupPosteriorsOutput'])
  writer = parsers.getTsvWriter(params['groupPosteriorsOutput'])
  
  writer.writerow(["protein", "group"] + ['%.4g' % x for x in params['proteinQuantCandidates']])
  for i, (_, protein, _, _, _, _, _, posteriorDists) in enumerate(proteinOutputRows):
    if posteriorDists:
      _, pProteinGroupQuants, _ = posteriorDists

      numGroups = len(params['groups'])
      for groupId, posterior in zip(range(numGroups), pProteinGroupQuants):
        writer.writerow([protein, params['groupLabels'][groupId]] + ['%.4g' % p for p in posterior])

def printFoldChangePosteriors(proteinOutputRows, params):
  print("Writing fold change posteriors to", params['foldChangePosteriorsOutput'])
  writer = parsers.getTsvWriter(params['foldChangePosteriorsOutput'])
  
  writer.writerow(["protein", "comparison"] + ['%.4g' % x for x in params['proteinDiffCandidates']])
  for i, (_, protein, _, _, _, _, _, posteriorDists) in enumerate(proteinOutputRows):
    if posteriorDists:
      _, _, pProteinGroupDiffs = posteriorDists
      numGroups = len(params['groups'])
      if numGroups >= 2:
        for groupId1, groupId2 in itertools.combinations(range(numGroups), 2):
          writer.writerow([protein, params['groupLabels'][groupId1] + "_vs_" + params['groupLabels'][groupId2]] + ['%.4g' % p for p in pProteinGroupDiffs[(groupId1, groupId2)]])

