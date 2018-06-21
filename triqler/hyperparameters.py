#!/usr/bin/python

from __future__ import print_function

import csv
import sys
import os
import itertools

import numpy as np
from scipy.stats import hypsecant, gamma, norm, binom
from scipy.optimize import curve_fit

from . import parsers

def fitPriors(peptQuantRows, params, printImputedVals = False, plot = False):
  params['proteinQuantCandidates'] = np.arange(-5.0, 5.0 + 1e-10, 0.01) # log10 of protein ratio  
  qc = params['proteinQuantCandidates']
  params['proteinDiffCandidates'] = np.linspace(2*qc[0], 2*qc[-1], len(qc)*2-1)
  
  protQuantRows = parsers.filterAndGroupPeptides(peptQuantRows, lambda x : not x.protein[0].startswith(params['decoyPattern']))
  
  imputedVals, imputedDiffs, observedXICValues, protQuants, protDiffs, protStdevsInGroup, protGroupDiffs = list(), list(), list(), list(), list(), list(), list()
  quantRowsCollection = list()
  for prot, quantRows in protQuantRows:
    quantRows, quantMatrix = parsers.getQuantMatrix(quantRows)
    
    quantMatrixNormalized = [parsers.geoNormalize(row) for row in quantMatrix]
    quantRowsCollection.append((quantRows, quantMatrix))
    geoAvgQuantRow = getProteinQuant(quantMatrixNormalized, quantRows)
    
    protQuants.extend([np.log10(x) for x in geoAvgQuantRow if not np.isnan(x)])

    args = parsers.getQuantGroups(geoAvgQuantRow, params["groups"], np.log10)
    means = list()
    for group in args:
      if np.count_nonzero(~np.isnan(group)) > 1:
        protDiffs.extend(group - np.mean(group))
        protStdevsInGroup.append(np.std(group))
      if np.count_nonzero(~np.isnan(group)) > 0:
        means.append(np.mean(group))
    
    if np.count_nonzero(~np.isnan(means)) > 1:
      for mean in means:  
        #protGroupDiffs.append(mean - np.mean(means))
        protGroupDiffs.append(mean)
    
    quantMatrixFiltered = np.log10(np.array([x for x, y in zip(quantMatrix, quantRows) if y.combinedPEP < 1.0]))  
    observedXICValues.extend(quantMatrixFiltered[~np.isnan(quantMatrixFiltered)])
    
    # counts number of NaNs per run, if there is only 1 non NaN in the column, we cannot use it for estimating the imputedDiffs distribution
    numNonNaNs = np.count_nonzero(~np.isnan(quantMatrixFiltered), axis = 0)[np.newaxis,:]
    xImps = imputeValues(quantMatrixFiltered, geoAvgQuantRow, np.log10(geoAvgQuantRow))
    imputedDiffs.extend((xImps - quantMatrixFiltered)[(~np.isnan(quantMatrixFiltered)) & (np.array(numNonNaNs) > 1)])
    #imputedVals.extend(xImps[(np.isnan(quantMatrixFiltered)) & (np.array(numNonNaNs) > 1)])
  
  fitLogitNormal(observedXICValues, params, plot)
  
  fitDist(protQuants, funcHypsec, "log10(protein ratio)", ["muProtein", "sigmaProtein"], params, plot)
    
  fitDist(imputedDiffs, funcHypsec, "log10(imputed xic / observed xic)", ["muFeatureDiff", "sigmaFeatureDiff"], params, plot)
  
  fitDist(protStdevsInGroup, lambda x, shape, sigma: gamma.pdf(x, shape, 0.0, sigma), "stdev log10(protein diff in group)", ["shapeInGroupStdevs", "scaleInGroupStdevs"], params, plot, x = np.arange(-0.1, 1.0, 0.005))
  
  sigmaCandidates = np.arange(0.001, 3.0, 0.001)
  gammaCandidates = gamma.pdf(sigmaCandidates, params["shapeInGroupStdevs"], 0.0, params["scaleInGroupStdevs"])
  support = np.where(gammaCandidates > max(gammaCandidates) * 0.01)
  params['sigmaCandidates'] = np.linspace(sigmaCandidates[support[0][0]], sigmaCandidates[support[0][-1]], 20)
  
  params['proteinPrior'] = funcHypsec(params['proteinQuantCandidates'], params["muProtein"], params["sigmaProtein"])
  if "shapeInGroupStdevs" in params:
    params['inGroupDiffPrior'] = funcHypsec(params['proteinDiffCandidates'], 0, params['sigmaCandidates'][:, np.newaxis])
  else: # if we have technical replicates, we could use a delta function for the group scaling parameter to speed things up
    fitDist(protDiffs, funcHypsec, "log10(protein diff in group)", ["muInGroupDiffs", "sigmaInGroupDiffs"], params, plot)
    params['inGroupDiffPrior'] = funcHypsec(params['proteinDiffCandidates'], params['muInGroupDiffs'], params['sigmaInGroupDiffs'])
  
  #fitDist(protGroupDiffs, funcHypsec, "log10(protein diff between groups)", ["muProteinGroupDiffs", "sigmaProteinGroupDiffs"], params, plot)
  
def fitLogitNormal(observedValues, params, plot):
  m = np.mean(observedValues)
  s = np.std(observedValues)
  minBin, maxBin = m - 4*s, m + 4*s
  #minBin, maxBin = -2, 6
  vals, bins = np.histogram(observedValues, bins = np.arange(minBin, maxBin, 0.1), normed = True)
  bins = bins[:-1]
  popt, pcov = curve_fit(funcLogitNormal, bins, vals, p0 = (m, s, m - s, s))
  #print("params[\"muDetectInit\"], params[\"sigmaDetectInit\"] = %f, %f" % (popt[0], popt[1]))
  print("params[\"muDetect\"], params[\"sigmaDetect\"] = %f, %f" % (popt[0], popt[1]))
  print("params[\"muXIC\"], params[\"sigmaXIC\"] = %f, %f" % (popt[2], popt[3]))
  #params["muDetectInit"], params["sigmaDetectInit"] = popt[0], popt[1]
  params["muDetect"], params["sigmaDetect"] = popt[0], popt[1]
  params["muXIC"], params["sigmaXIC"] = popt[2], popt[3]
  if plot:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.bar(bins, vals, width = bins[1] - bins[0], alpha = 0.5)
    plt.plot(bins, norm.pdf(bins, np.mean(observedValues), np.std(observedValues)), label='normal fit')
    plt.plot(bins, funcLogitNormal(bins, *popt), label='logit-normal fit')
    plt.plot(bins, 0.5 + 0.5 * np.tanh((np.array(bins) - popt[0]) / popt[1]), label = "detection ratio")
    plt.plot(bins, norm.pdf(bins, popt[2], popt[3]), label = "underlying distribution")
    plt.xlabel("log10(intensity)")
    plt.legend()
    
def fitDist(ys, func, xlabel, varNames, params, plot, x = np.arange(-2,2,0.01)):
  vals, bins = np.histogram(ys, bins = x, normed = True)
  bins = bins[:-1]
  popt, pcov = curve_fit(func, bins, vals)
  outputString = ", ".join(["params[\"%s\"]"]*len(popt)) + " = " + ", ".join(["%f"] * len(popt))
  for varName, val in zip(varNames, popt):
    params[varName] = val
  print(outputString % tuple(varNames + list(popt)))
  if plot:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.bar(bins, vals, width = bins[1] - bins[0])
    plt.plot(bins, func(bins, *popt), 'g', label='distribution fit')
    plt.xlabel(xlabel)
    plt.legend()

# this is an optimized version of applying parsers.weightedGeomAvg to each of the columns separately
def getProteinQuant(quantMatrixNormalized, quantRows):
  numSamples = len(quantMatrixNormalized[0])
  
  geoAvgQuantRow = np.array([0.0]*numSamples)
  weights = np.array([[1.0 - y.combinedPEP for y in quantRows]] * numSamples).T
  weights[np.isnan(np.array(quantMatrixNormalized))] = np.nan
  
  weightSum = np.nansum(weights, axis = 0)
  weightSum[weightSum == 0] = np.nan
  geoAvgQuantRow = np.exp(np.nansum(np.multiply(np.log(quantMatrixNormalized), weights), axis = 0) / weightSum)
  geoAvgQuantRow = parsers.geoNormalize(geoAvgQuantRow)
  return geoAvgQuantRow
  
def imputeValues(quantMatrixLog, proteinRatios, testProteinRatios):
  logIonizationEfficiencies = quantMatrixLog - np.log10(proteinRatios)
  
  numNonZeros = np.count_nonzero(~np.isnan(logIonizationEfficiencies), axis = 1)[:,np.newaxis] - ~np.isnan(logIonizationEfficiencies)
  np.nan_to_num(logIonizationEfficiencies, False)
  meanLogIonEff = (np.nansum(logIonizationEfficiencies, axis = 1)[:,np.newaxis] - logIonizationEfficiencies) / numNonZeros
  
  logImputedVals = meanLogIonEff + testProteinRatios
  return logImputedVals
    
def funcLogitNormal(x, muLogit, sigmaLogit, muNorm, sigmaNorm):
  return logit(x, muLogit, sigmaLogit) * norm.pdf(x, muNorm, sigmaNorm)

def funcNorm(x, mu, sigma):
  return norm.pdf(x, mu, sigma)
  
def funcHypsec(x, mu, sigma):
  return hypsecant.pdf(x, mu, sigma)

def logit(x, muLogit, sigmaLogit):
  return 0.5 + 0.5 * np.tanh((np.array(x) - muLogit) / sigmaLogit)

