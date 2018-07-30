#!/usr/bin/python

from __future__ import print_function

import itertools

import numpy as np
from scipy.stats import f_oneway, gamma
from scipy.optimize import curve_fit

from . import parsers
from . import convolution_dp
from . import hyperparameters

def getPosteriors(quantRowsOrig, params, returnDistributions = False):
  quantRows, quantMatrix = parsers.getQuantMatrix(quantRowsOrig)
  
  pProteinQuantsList, bayesQuantRow = getPosteriorProteinRatios(quantMatrix, quantRows, params)
  pProteinGroupQuants = getPosteriorProteinGroupRatios(pProteinQuantsList, bayesQuantRow, params)
  pProteinGroupDiffs, muGroupDiffs = getProteinGroupsDiffPosteriors(pProteinGroupQuants, params)
  
  probsBelowFoldChange = getProbBelowFoldChangeDict(pProteinGroupDiffs, params)
  if returnDistributions:
    return bayesQuantRow, muGroupDiffs, probsBelowFoldChange, pProteinQuantsList, pProteinGroupQuants, pProteinGroupDiffs
  else:
    return bayesQuantRow, muGroupDiffs, probsBelowFoldChange

def getPosteriorProteinRatios(quantMatrix, quantRows, params, maxIterations = 50, bayesQuantRow = None):
  numSamples = len(quantMatrix[0])
  bayesQuantRow = np.array([1.0]*numSamples)
  for iteration in range(maxIterations):  
    prevBayesQuantRow = np.copy(bayesQuantRow)
    pProteinQuantsList, bayesQuantRow = getPosteriorProteinRatio(quantMatrix, quantRows, bayesQuantRow, params)
    
    bayesQuantRow = parsers.geoNormalize(bayesQuantRow)
    
    diffInIteration = np.log10(prevBayesQuantRow) - np.log10(bayesQuantRow)
    if np.max(diffInIteration*diffInIteration) < 1e-4:
      #print("Converged after iteration", iteration+1)
      break
  
  return pProteinQuantsList, bayesQuantRow

def getPosteriorProteinRatio(quantMatrix, quantRows, geoAvgQuantRow, params):
  numSamples = len(quantMatrix[0])
  
  logGeoAvgs = np.log10([parsers.geomAvg(row) for row in quantMatrix])
  featDiffs = np.log10(quantMatrix) - logGeoAvgs[:,np.newaxis]
  pMissingGeomAvg = pMissing(logGeoAvgs, params["muDetect"], params["sigmaDetect"]) # Pr(f_grn = NaN | t_grn = 1)
  
  pQuantIncorrectId = hyperparameters.funcHypsec(featDiffs, params["muFeatureDiff"], params["sigmaFeatureDiff"]) # Pr(f_grn = x | t_grn = 1)
  #pQuantIncorrectIdOld = hyperparameters.funcLogitNormal(np.log10(quantMatrix), params["muDetect"], params["sigmaDetect"], params["muXIC"], params["sigmaXIC"]) 
  
  xImpsAll = imputeValues(quantMatrix, geoAvgQuantRow, params['proteinQuantCandidates'])
  impDiffs = xImpsAll - np.log10(np.array(quantMatrix))[:,:,np.newaxis]
  pDiffs = hyperparameters.funcHypsec(impDiffs, params["muFeatureDiff"], params["sigmaFeatureDiff"]) # Pr(f_grn = x | m_grn = 0, t_grn = 0)
  
  pProteinQuantsList, bayesQuantRow = list(), list()
  for j in range(numSamples):    
    pProteinQuant = np.log(params['proteinPrior']) # log likelihood
    #pProteinQuant = np.zeros(len(params['proteinPrior']))
    
    for i, row in enumerate(quantMatrix):
      linkPEP = quantRows[i].linkPEP[j]
      identPEP = quantRows[i].identificationPEP[j]
      if identPEP < 1.0:
        pMissings = pMissing(xImpsAll[i,j,:], params["muDetect"], params["sigmaDetect"]) # Pr(f_grn = NaN | m_grn = 1, t_grn = 0)
        if np.isnan(row[j]):
          likelihood = pMissings * (1.0 - identPEP) * (1.0 - linkPEP) + pMissingGeomAvg[i] * (identPEP * (1.0 - linkPEP) + linkPEP)
        else:
          likelihood = (1.0 - pMissings) * pDiffs[i,j,:] * (1.0 - identPEP) * (1.0 - linkPEP) + (1.0 - pMissingGeomAvg[i]) * (pQuantIncorrectId[i][j] * identPEP * (1.0 - linkPEP) + linkPEP)
        pProteinQuant += np.log(likelihood)
    pProteinQuant = np.exp(pProteinQuant) / np.sum(np.exp(pProteinQuant))
    
    pProteinQuantsList.append(pProteinQuant)
    
    eValue, confRegion = getPosteriorParams(params['proteinQuantCandidates'], pProteinQuant)
    bayesQuantRow.append(eValue)
  
  return pProteinQuantsList, bayesQuantRow

def imputeValues(quantMatrix, proteinRatios, testProteinRatios):
  logIonizationEfficiencies = np.log10(quantMatrix) - np.log10(proteinRatios)
  
  numNonZeros = np.count_nonzero(~np.isnan(logIonizationEfficiencies), axis = 1)[:,np.newaxis] - ~np.isnan(logIonizationEfficiencies)
  np.nan_to_num(logIonizationEfficiencies, False)
  meanLogIonEff = (np.nansum(logIonizationEfficiencies, axis = 1)[:,np.newaxis] - logIonizationEfficiencies) / numNonZeros
  
  logImputedVals = np.tile(meanLogIonEff[:, :, np.newaxis], (1, 1, len(testProteinRatios))) + testProteinRatios
  return logImputedVals

def pMissing(x, muLogit, sigmaLogit):
  return 1.0 - hyperparameters.logit(x, muLogit, sigmaLogit) + np.nextafter(0, 1)

def getPosteriorProteinGroupRatios(pProteinQuantsList, bayesQuantRow, params):
  numGroups = len(params["groups"])
  
  pProteinGroupQuants = list()
  for groupId in range(numGroups):
    filteredProteinQuantsList = np.array([x for j, x in enumerate(pProteinQuantsList) if j in params['groups'][groupId]])
    if "shapeInGroupStdevs" in params:
      pMu = getPosteriorProteinGroupMuMarginalized(filteredProteinQuantsList, params)
    else:
      pMu = getPosteriorProteinGroupMu(params['inGroupDiffPrior'], filteredProteinQuantsList, params)
    pProteinGroupQuants.append(pMu)
  
  return pProteinGroupQuants
  
def getPosteriorProteinGroupMu(pDiffPrior, pProteinQuantsList, params):
  pMus = np.zeros_like(params['proteinQuantCandidates'])
  for pProteinQuants in pProteinQuantsList:
    pMus += np.log(np.convolve(pDiffPrior, pProteinQuants, mode = 'valid'))
  
  #pMus = np.nan_to_num(pMus)
  #pMus -= np.max(pMus)
  pMus = np.exp(pMus) / np.sum(np.exp(pMus))
  return pMus

def getPosteriorProteinGroupMuMarginalized(pProteinQuantsList, params):
  pMus = np.zeros((len(params['sigmaCandidates']), len(params['proteinQuantCandidates'])))
  for pProteinQuants in pProteinQuantsList:
    for idx, pDiffPrior in enumerate(params['inGroupDiffPrior']):
      pMus[idx,:] += np.log(np.convolve(pDiffPrior, pProteinQuants, mode = 'valid'))
  
  pSigmas = gamma.pdf(params['sigmaCandidates'], params["shapeInGroupStdevs"], 0.0, params["scaleInGroupStdevs"]) # prior
  pMus = np.log(np.dot(pSigmas, np.exp(pMus)))
  
  pMus = np.exp(pMus) / np.sum(np.exp(pMus))
  
  return pMus
  
def getProteinGroupsDiffPosteriors(pProteinGroupQuants, params):
  numGroups = len(params['groups'])  
  pProteinGroupDiffs, muGroupDiffs = dict(), dict()
  for groupId1, groupId2 in itertools.combinations(range(numGroups), 2):
    pDifference = np.convolve(pProteinGroupQuants[groupId1], pProteinGroupQuants[groupId2][::-1])
    pProteinGroupDiffs[(groupId1,groupId2)] = pDifference
    muGroupDiffs[(groupId1,groupId2)], _ = np.log2(getPosteriorParams(params['proteinDiffCandidates'], pDifference) + np.nextafter(0, 1))
  return pProteinGroupDiffs, muGroupDiffs
  
def getProbBelowFoldChangeDict(pProteinGroupDiffs, params):
  probsBelowFoldChange = dict()
  numGroups = len(params["groups"])
  for groupId1, groupId2 in itertools.combinations(range(numGroups), 2):
    probsBelowFoldChange[(groupId1, groupId2)] = getPosteriorProteinGroupDiff(pProteinGroupDiffs[(groupId1, groupId2)], params)
  #probsBelowFoldChange['ANOVA'] = getProbBelowFoldChangeANOVA(pProteinGroupQuants, params)
  return probsBelowFoldChange

def getPosteriorProteinGroupDiff(pDifference, params):  
  return sum([y for x, y in zip(params['proteinDiffCandidates'], pDifference) if abs(np.log2(10**x)) < params['foldChangeEval']])

# this is a "pseudo"-ANOVA test which calculates the probability distribution 
# for differences of means between multiple groups. With <=4 groups it seemed
# to return reasonable results, but with 10 groups it called many false positives.
def getProbBelowFoldChangeANOVA(pProteinGroupQuants, params):
  if len(pProteinGroupQuants) > 4:
    print("WARNING: this ANOVA-like test might not behave well if >4 treatment groups are present")
  
  if len(pProteinGroupQuants) >= 2:
    convProbs = convolution_dp.convolveProbs(pProteinGroupQuants)
    bandwidth = np.searchsorted(params['proteinQuantCandidates'], params['proteinQuantCandidates'][0] + np.log10(2**params['foldChangeEval']))
    probBelowFoldChange = 0.0
    for i in range(bandwidth):
      probBelowFoldChange += np.trace(convProbs, offset = i)
  else:
    probBelowFoldChange = 1.0
  return min([1.0, probBelowFoldChange])
  
def getPosteriorParams(proteinQuantCandidates, pProteinQuants):
  return 10**np.sum(proteinQuantCandidates * pProteinQuants), 0.0
  if False:
    eValue, variance = 0.0, 0.0
    for proteinRatio, pq in zip(proteinQuantCandidates, pProteinQuants):
      if pq > 0.001:
        #print(10**proteinRatio, pq)
        eValue += proteinRatio * pq

    for proteinRatio, pq in zip(proteinQuantCandidates, pProteinQuants):
      if pq > 0.001:
        variance += pq * (proteinRatio - eValue)**2
    eValueNew = 10**eValue
    
    return eValueNew, [10**(eValue - np.sqrt(variance)), 10**(eValue + np.sqrt(variance))]

