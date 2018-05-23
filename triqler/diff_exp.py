#!/usr/bin/python

from __future__ import print_function

import csv
import itertools

import numpy as np
from scipy.stats import f_oneway, kruskal

from . import percolator
from . import parsers

def doDiffExp(params, peptQuantRows, peptQuantRowFile, proteinQuantificationMethod, selectComparison, qvalMethod):    
  proteinModifier, getEvalFeatures, evalFunctions = getEvalFunctions(peptQuantRowFile, params)
  
  proteinOutputRows = proteinQuantificationMethod(peptQuantRows, params, proteinModifier, getEvalFeatures)
  
  numGroups = len(params['groups'])
  if numGroups > 2:
    for groupId1, groupId2 in itertools.combinations(range(numGroups), 2):
      params['groupIdsDiffExp'] = (groupId1, groupId2)
      proteinOutputFile = peptQuantRowFile.replace("_peptides.tsv", "_proteins.%dvs%d.tsv" % (groupId1 + 1, groupId2 + 1))
      print(proteinOutputFile)
      if len(params["trueConcentrationsDict"]) > 0:
        evalFunctions = [lambda protein, evalFeatures : evalTruePositiveTtest(params["trueConcentrationsDict"], protein, groupId1, groupId2, evalFeatures[-2], params)]
      proteinOutputRowsGroup = selectComparison(proteinOutputRows, (groupId1, groupId2))
      getQvals(proteinOutputRowsGroup, qvalMethod = qvalMethod, evalFunctions = evalFunctions, outputFile = proteinOutputFile, params = params)
  
  proteinOutputFile = peptQuantRowFile.replace("_peptides.tsv", "_proteins.tsv")
  print(proteinOutputFile)
  proteinOutputRowsGroup = selectComparison(proteinOutputRows, 'ANOVA')
  if len(params["trueConcentrationsDict"]) > 0:
    evalFunctions = [lambda protein, evalFeatures : evalTruePositiveANOVA(params["trueConcentrationsDict"], protein)]
  getQvals(proteinOutputRowsGroup, qvalMethod = qvalMethod, evalFunctions = evalFunctions, outputFile = proteinOutputFile, params = params)

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
  
def getQvals(peptideOutputRows, qvalMethod, evalFunctions, outputFile, params):
  writer = csv.writer(open(outputFile, 'w'), delimiter = '\t')
  plotCalibration = len(evalFunctions) > 0
  if plotCalibration:
    evalTruePositives = evalFunctions[0]
  
  evalHeaders = ["log2_fold_change", "diff_exp_pval_" + str(params['foldChangeEval'])]
  outRows = list()
  observedQvals, reportedQvals, reportedPEPs = list(), list(), list()
  sumPEP, fp, tp = 0.0, 1, 0
  decoys, targets = 1, 0
  numTies = 1
  
  if 'pvalues' in qvalMethod:
    targetPvalues = list()
    for i, (combinedPEP, _, protein, quantRows, evalFeatures, numPeptides, proteinIdPEP) in enumerate(peptideOutputRows):
      targetPvalues.append(evalFeatures[-1])
    reportedQvalsPval, reportedPEPsPval = percolator.getQvalues(targetPvalues, includePEPs = True)
  
  nextScores = [x[0] for x in peptideOutputRows] + [np.nan]
  for i, (combinedPEP, _, protein, quantRows, evalFeatures, numPeptides, proteinIdPEP) in enumerate(peptideOutputRows):
    if 'pvalues_with_fc' in qvalMethod and np.abs(evalFeatures[-2]) < params['foldChangeEval']:
      continue
    
    if plotCalibration:
      if not "decoy_" in protein:
        if evalTruePositives(protein, evalFeatures):
          tp += 1
        else:
          fp += 1
      observedQval = float(fp) / (tp+fp)

    score = combinedPEP
    if not "decoy_" in protein:
      sumPEP += combinedPEP
      targets += 1
      qval = sumPEP / targets

    if score == nextScores[i+1]:
      numTies += 1
    else:
      for _ in range(numTies):
        if 'pvalues' in qvalMethod:
          reportedQvals.append(reportedQvalsPval[i])
          reportedPEPs.append(reportedPEPsPval[i])
        else:
          reportedQvals.append(qval)
        
        if plotCalibration:
          observedQvals.append(observedQval)
      numTies = 1

    outRows.append([combinedPEP, protein, numPeptides, proteinIdPEP] + evalFeatures + [quantRows[0].peptide, quantRows[0].identificationPEP])

  if 'pvalues' not in qvalMethod:
    reportedPEPs = [np.nan]*len(reportedQvals)
  
  if plotCalibration:
    observedQvals = fdrsToQvals(observedQvals)

    writer.writerow(["observed_qval", "reported_qval", "reported_PEP", "combined_PEP", "protein", "num_peptides", "protein_id_PEP"] + evalHeaders + ["peptide", "peptide_PEP"])
    for outRow, observedQval, reportedQval, reportedPEP in zip(outRows, observedQvals, reportedQvals, reportedPEPs):
      writer.writerow(["%.4g" % (observedQval), "%.4g" % (reportedQval), "%.4g" % (reportedPEP)] + outRow)
  else:
    writer.writerow(["qval", "PEP", "combined_PEP", "protein", "num_peptides", "protein_id_PEP"] + evalHeaders + ["peptide", "peptide_PEP"])
    for outRow, reportedQval, reportedPEP in zip(outRows, reportedQvals, reportedPEPs):
      writer.writerow(["%.4g" % (reportedQval), "%.4g" % (reportedPEP)] + outRow)

def fdrsToQvals(fdrs):
  qvals = [0] * len(fdrs)
  if len(fdrs) > 0:
    qvals[len(fdrs)-1] = fdrs[-1]
    for i in range(len(fdrs)-2, -1, -1):
      qvals[i] = min(qvals[i+1], fdrs[i])
  return qvals
