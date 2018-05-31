#!/usr/bin/python

from __future__ import print_function

import itertools

import numpy as np
from scipy.stats import f_oneway, gamma
from scipy.optimize import curve_fit

from . import parsers
from . import convolution_dp
from . import hyperparameters

def getPosteriors(quantRowsOrig, params):
  quantRows, quantMatrix = parsers.getQuantMatrix(quantRowsOrig)
  pProteinQuantsList, bayesQuantRow = getPosteriorProteinRatios(quantMatrix, quantRows, params)
  pProteinGroupQuants, mus, sigmas = getPosteriorProteinGroupRatios(pProteinQuantsList, bayesQuantRow, params)
  #probsBelowFoldChange = getPosteriorProteinGroupsDiffs(pProteinGroupQuants, params)
  probsBelowFoldChange = getProbBelowFoldChangeDict(pProteinGroupQuants, params)
  #mus, sigmas, probBelowFoldChange = list(), list(), 1.0
  return bayesQuantRow, mus, sigmas, probsBelowFoldChange
  
def getProbBelowFoldChangeDict(pProteinGroupQuants, params):
  probsBelowFoldChange = dict()
  numGroups = len(params["groups"])
  if numGroups > 2:
    for groupId1, groupId2 in itertools.combinations(range(numGroups), 2):
      probsBelowFoldChange[(groupId1, groupId2)] = getPosteriorProteinGroupDiff(pProteinGroupQuants, groupId1, groupId2, params)
  probsBelowFoldChange['ANOVA'] = getProbBelowFoldChange(pProteinGroupQuants, params)
  return probsBelowFoldChange

def getProbBelowFoldChange(pProteinGroupQuants, params):  
  if len(pProteinGroupQuants) >= 2:
    convProbs = convolution_dp.convolveProbs(pProteinGroupQuants)
    bandwidth = np.searchsorted(params['proteinQuantCandidates'], params['proteinQuantCandidates'][0] + np.log10(2**params['foldChangeEval']))
    probBelowFoldChange = 0.0
    for i in range(bandwidth):
      probBelowFoldChange += np.trace(convProbs, offset = i)
  else:
    probBelowFoldChange = 1.0
  return min([1.0, probBelowFoldChange])
      
def printQuantRows(quantMatrix, quantRows):
  for i, row in enumerate(quantMatrix):
    print("\t".join(['%.2f' % x for x in quantMatrix[i]]) + '\tcombinedPEP=' + '%.2g' % quantRows[i].combinedPEP)
  print("")
  
def printStats(geoAvgQuantRow, groups):
  print("\t".join(['%.2f' % x for x in geoAvgQuantRow]))
  print("")
  
  args = parsers.getQuantGroups(geoAvgQuantRow, groups)
  anovaFvalue, anovaPvalue = f_oneway(*args)
  print("p-value:", anovaPvalue)
  print("")
  
  float_formatter = lambda x: "%.2f" % x
  np.set_printoptions(formatter={'float_kind':float_formatter})
  
  geoAvgs = np.matrix([parsers.geomAvg([2**y for y in x]) for x in args])
  ratioMatrix = np.log2(np.transpose(geoAvgs) / geoAvgs)
  print(ratioMatrix)
  print("")

def getPosteriorProteinGroupRatios(pProteinQuantsList, bayesQuantRow, params, fitCurves = False):
  args = parsers.getQuantGroups(bayesQuantRow, params["groups"], np.log10)
  
  muProteinGroup = [np.mean(x) for x in args]
  numGroups = len(params["groups"])
  #sigmaProteinDiff = [np.std(x) for x in args]
  #print([np.std(x) for x in args])
  sigmaProteinDiff = [params['sigmaProteinDiffs'] for _ in range(numGroups)]
  
  #if "shapeProteinStdevs" in params:
  #  pSigma = gamma.pdf(params['sigmaCandidates'], params["shapeProteinStdevs"], 0.0, params["scaleProteinStdevs"])
  
  pProteinGroupQuants = [[0.0] * len(params['proteinQuantCandidates']) for _ in range(numGroups)]
  mus, sigmas = list(), list()
  for groupId in range(numGroups):
    if "shapeProteinStdevs" in params:
      filteredPProteinQuantsList = np.array([x for j, x in enumerate(pProteinQuantsList) if j in params['groups'][groupId]])
      pSigma = getPosteriorProteinGroupSigma(muProteinGroup[groupId], groupId, filteredPProteinQuantsList, params)
      #sigmaProteinDiff[groupId] = sigmaCandidates[np.argmax(pSigma)]
      pProteinGroupQuants[groupId] = getPosteriorProteinGroupMuSigmaPrior(pSigma, groupId, filteredPProteinQuantsList, params)
    else:
      if sigmaProteinDiff[groupId] == 0.0:
        sigmaProteinDiff[groupId] = np.nanmean(sigmaProteinDiff)  
      pProteinGroupQuants[groupId] = getPosteriorProteinGroupMu(sigmaProteinDiff[groupId], groupId, pProteinQuantsList, params)
    
    # fit protein group ratios with normal distribution
    if fitCurves:
      popt, pcov = curve_fit(hyperparameters.funcNorm, params['proteinQuantCandidates'], pProteinGroupQuants[groupId] * 100)
      mu, sigma = popt
      mus.append(mu)
      sigmas.append(sigma)
      
      varNames = ["mu", "sigma"]    
      outputString = ", ".join(["params[\"%s\"]"]*len(popt)) + " = " + ", ".join(["%f"] * len(popt))
      print(outputString % tuple(varNames + list(popt)))
  
  # fit protein ratios with normal distribution
  if fitCurves and False:
    for pProteinQuants in pProteinQuantsList:
      popt, pcov = curve_fit(hyperparameters.funcNorm, params['proteinQuantCandidates'], pProteinQuants * 100)
      mu, sigma = popt
      mus.append(mu)
      sigmas.append(sigma)
  return pProteinGroupQuants, mus, sigmas

def getPosteriorProteinGroupDiff(pProteinGroupQuants, groupId1, groupId2, params):
  pDifference = np.convolve(pProteinGroupQuants[groupId1], pProteinGroupQuants[groupId2][::-1])
  difference = np.arange(-10, 10 + 1e-10, 0.01)
  
  return sum([y for x, y in zip(difference, pDifference) if abs(np.log2(10**x)) < params['foldChangeEval']])
  
def getProteinGroupsDiffPosteriors(pProteinGroupQuants, params):
  numGroups = len(params['groups'])  
  groupDiffPosteriors = dict()
  for groupId1, groupId2 in itertools.combinations(range(numGroups), 2):
    pDifference = np.convolve(pProteinGroupQuants[groupId1], pProteinGroupQuants[groupId2][::-1])
    difference = np.arange(-10, 10 + 1e-10, 0.01)
    
    groupDiffPosteriors[(groupId1,groupId2)] = (difference, pDifference)
  return groupDiffPosteriors

def getPosteriorProteinRatios(quantMatrix, quantRows, params, printStats = False, maxIterations = 50, bayesQuantRow = None):
  numSamples = len(quantMatrix[0])
  bayesQuantRow = np.array([1.0]*numSamples)
  for iteration in range(maxIterations):  
    prevBayesQuantRow = np.copy(bayesQuantRow)
    pProteinQuantsList, bayesQuantRow = getPosteriorProteinRatio(quantMatrix, quantRows, bayesQuantRow, params, printStats = printStats)
    
    bayesQuantRow = parsers.geoNormalize(bayesQuantRow)
    
    if printStats:
      printStats(bayesQuantRow, params['groups'])
    
    diffInIteration = np.log10(prevBayesQuantRow) - np.log10(bayesQuantRow)
    if np.max(diffInIteration*diffInIteration) < 1e-4:
      #print("Converged after iteration", iteration+1)
      break
  return pProteinQuantsList, bayesQuantRow

#@profile
def getPosteriorProteinRatio(quantMatrix, quantRows, geoAvgQuantRow, params, printStats = False):
  numSamples = len(quantMatrix[0])
  
  bayesQuantRow = np.array([1.0]*numSamples)
  
  #pQuantIncorrectIdOld = hyperparameters.funcLogitNormal(np.log10(quantMatrix), params["muDetect"], params["sigmaDetect"], params["muXIC"], params["sigmaXIC"]) # f_grn = x | t_grn = 1
  pQuantIncorrectId = hyperparameters.funcHypsec(np.log10(quantMatrix) - np.log10([parsers.geomAvg(row) for row in quantMatrix])[:,np.newaxis], params["muFeatureDiff"], params["sigmaFeatureDiff"])
  xImpsAll = imputeValues(quantMatrix, geoAvgQuantRow, params['proteinQuantCandidates'])
  impDiffs = xImpsAll - np.log10(np.array(quantMatrix))[:,:,np.newaxis]
  pDiffs = hyperparameters.funcHypsec(impDiffs, params["muFeatureDiff"], params["sigmaFeatureDiff"]) # f_grn = x | m_grn = 0, t_grn = 0
  plot = False
  pProteinQuantsList = list()
  for j in range(numSamples):
    #priorProteinQuantCandidates = hyperparameters.funcHypsec(params['proteinQuantCandidates'], np.log10(geoAvgQuantRow[j]), params["sigmaProtein"])
    priorProteinQuantCandidates = hyperparameters.funcHypsec(params['proteinQuantCandidates'], params["muProtein"], params["sigmaProtein"])
    
    pProteinQuant = np.log(priorProteinQuantCandidates) # log likelihood
    
    for i, row in enumerate(quantMatrix):
      linkPEP = quantRows[i].linkPEP[j]
      if linkPEP < 1.0:
        xImps = xImpsAll[i,j,:] # imputeValue(row, geoAvgQuantRow, params['proteinQuantCandidates'], j)
        pMissings = pMissingLogInput(xImps, params["muDetect"], params["sigmaDetect"]) # f_grn = NaN | m_grn = 1, t_grn = 0
        if np.isnan(row[j]):
          pMissingIncorrectId = pMissing(parsers.geomAvg(row), params["muDetect"], params["sigmaDetect"]) # f_grn = NaN | t_grn = 1
          likelihood = pMissings * (1.0 - linkPEP) + pMissingIncorrectId * linkPEP
        else:
          likelihood = (1.0 - pMissings) * pDiffs[i,j,:] * (1.0 - linkPEP) + pQuantIncorrectId[i][j] * linkPEP
        pProteinQuant += np.log(likelihood)
    pProteinQuants = np.exp(pProteinQuant) / np.sum(np.exp(pProteinQuant))
    
    pProteinQuantsList.append(pProteinQuants)
    
    eValue, confRegion = getPosteriorParams(params['proteinQuantCandidates'], pProteinQuants)
    bayesQuantRow[j] = eValue
    
    if printStats:
      print(eValue, confRegion)
  
  #print(bayesQuantRow)
  if printStats:
    print("")
  
  return pProteinQuantsList, bayesQuantRow
  
def getPosteriorProteinGroupMu(sigmaProteinDiff, groupId, pProteinQuantsList, params):
  #priorDiffsOld = norm.pdf(params['proteinQuantCandidates'], params['proteinQuantCandidates'][:, np.newaxis], sigmaProteinDiff)
  qc = params['proteinQuantCandidates']
  diffCandidates = np.linspace(2*qc[0], 2*qc[-1], len(qc)*2-1)
  priorDiffs = hyperparameters.funcHypsec(diffCandidates, params['muProteinDiffs'], sigmaProteinDiff)
  pMus = np.zeros_like(params['proteinQuantCandidates'])
  for j, pProteinQuants in enumerate(pProteinQuantsList): 
    if len([g for g, x in enumerate(params['groups']) if j in x]) > 0:
      groupIdTest = [g for g, x in enumerate(params['groups']) if j in x][0]
    else:
      continue
    
    if groupId == groupIdTest:
      pMus += np.log(np.convolve(priorDiffs, pProteinQuants, mode = 'valid'))
      #pMus += np.log(np.sum(priorDiffsOld * pProteinQuants, axis = 1))
  
  pMus = np.nan_to_num(pMus)
  pMus -= max(pMus)
  pMus = np.exp(pMus) / np.sum(np.exp(pMus))
  return pMus

def getPosteriorProteinGroupMuSigmaPrior(pSigma, groupId, pProteinQuantsList, params):
  #priorDiffsOld = norm.pdf(params['proteinQuantCandidates'], params['proteinQuantCandidates'][:, np.newaxis], sigmaProteinDiff)
  priorDiffs = np.squeeze(np.dot(params['priorDiffsVarSigma'].T, pSigma[:, np.newaxis]))
  
  pMus = np.zeros_like(params['proteinQuantCandidates'])
  for pProteinQuants in pProteinQuantsList: 
    pMus += np.log(np.convolve(priorDiffs, pProteinQuants, mode = 'valid'))
    #pMus += np.log(np.sum(priorDiffsOld * pProteinQuants, axis = 1))
  
  pMus = np.nan_to_num(pMus)
  pMus -= max(pMus)
  pMus = np.exp(pMus) / np.sum(np.exp(pMus))
  return pMus

def getPosteriorProteinGroupSigma(muProteinGroup, groupId, pProteinQuantsList, params):
  pSigmas = np.log(gamma.pdf(params['sigmaCandidates'], params["shapeProteinStdevs"], 0.0, params["scaleProteinStdevs"])) # prior
  pGroupQuants = hyperparameters.funcNorm(params['proteinQuantCandidates'], muProteinGroup, params['sigmaCandidates'][:,np.newaxis])
  pProteinQuantPriors = np.dot(pProteinQuantsList, pGroupQuants.T)
  for pProteinQuants in pProteinQuantPriors:
    pSigmas += np.log(pProteinQuants)
  pSigmas = np.exp(pSigmas) / np.sum(np.exp(pSigmas))
  
  return pSigmas

def getProteinSigmaPosterior(protQuants):
  sigmaCandidates = np.arange(0.0001, 0.1, 0.0001)
  logLikelihoods = list()
  for sigma in sigmaCandidates:
    logLikelihoods.append(np.sum(np.log(hyperparameters.funcHypsec(protQuants, 0.0, sigma))))
  
  logLikelihoods = np.nan_to_num(logLikelihoods)
  logLikelihoods -= np.max(logLikelihoods)
  logLikelihoods = np.exp(logLikelihoods) / np.sum(np.exp(logLikelihoods))
  
  print("sigmaProtein MAP estimate:", sigmaCandidates[np.argmax(logLikelihoods)])
  
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

def imputeValues(quantMatrix, proteinRatios, testProteinRatios):
  logIonizationEfficiencies = np.log10(quantMatrix) - np.log10(proteinRatios)
  
  numNonZeros = np.count_nonzero(~np.isnan(logIonizationEfficiencies), axis = 1)[:,np.newaxis] - ~np.isnan(logIonizationEfficiencies)
  np.nan_to_num(logIonizationEfficiencies, False)
  meanLogIonEff = (np.nansum(logIonizationEfficiencies, axis = 1)[:,np.newaxis] - logIonizationEfficiencies) / numNonZeros
  
  logImputedVals = np.tile(meanLogIonEff[:, :, np.newaxis], (1, 1, len(testProteinRatios))) + testProteinRatios
  return logImputedVals
  
def pMissing(x, muLogit, sigmaLogit):
  return pMissingLogInput(np.log10(x), muLogit, sigmaLogit)

def pMissingLogInput(x, muLogit, sigmaLogit):
  return 1.0 - hyperparameters.logit(x, muLogit, sigmaLogit) + np.nextafter(0, 1)

