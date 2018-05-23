#!/usr/bin/python

from __future__ import print_function

import csv
import sys
import os
import itertools

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import hypsecant, gamma, norm, binom
from scipy.optimize import curve_fit

from . import parsers

def main(argv):
  peptInputFile = argv[0]
  
  matplotlib.rcParams['axes.unicode_minus'] = False
  
  params = dict()
  params["runIds"], params['groups'], params['groupLabels'], peptQuantRows = parsers.parsePeptideQuantFile(peptInputFile)  
  
  fitPriors(peptQuantRows, params, plot = True)
  plt.show()

def printQuantRows(quantMatrix, quantRows):
  for i, row in enumerate(quantMatrix):
    print("\t".join(['%.2f' % x for x in quantMatrix[i]]) + '\tcombinedPEP=' + '%.2g' % quantRows[i].combinedPEP)
  print("")

def fitPriors(peptQuantRows, params, printImputedVals = False, plot = False):
  params['proteinQuantCandidates'] = np.arange(-5.0, 5.0 + 1e-10, 0.01) # log10 of protein ratio
  params['sigmaCandidates'] = np.arange(0.01, 1.0, 0.01)
  
  peptQuantRows = filter(lambda x : "decoy_" not in x.protein and "NA" not in x.protein and x.protein.count(",") == 0 and x.combinedPEP < 1.0, peptQuantRows)
  
  protQuantRows = itertools.groupby(sorted(peptQuantRows, key = lambda x : x.protein), key = lambda x : x.protein)
  
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
    
    #if len(xImps[(np.isnan(quantMatrixFiltered)) & (np.array(numNonNaNs) > 1)]) > 0 and np.max(xImps[(np.isnan(quantMatrixFiltered)) & (np.array(numNonNaNs) > 1)]) > 3.0:
    #  printQuantRows(quantMatrix, quantRows)
  
  fitLogitNormal(observedXICValues, params, plot)
  
  #plt.figure()
  #plt.hist(imputedVals, bins = np.arange(-2,6,0.1))
  #plt.hist(observedXICValues, bins = np.arange(-2,6,0.1), alpha = 0.5)
  # update logit with non-parametric estimate
  #updateMissingLogit(observedXICValues, imputedVals, params, plot)
  
  fitDist(protQuants, funcHypsec, "log10(protein ratio)", ["muProtein", "sigmaProtein"], params, plot)
    
  fitDist(imputedDiffs, funcHypsec, "log10(imputed xic / observed xic)", ["muFeatureDiff", "sigmaFeatureDiff"], params, plot)
  
  fitDist(protDiffs, funcHypsec, "log10(protein diff in group)", ["muProteinDiffs", "sigmaProteinDiffs"], params, plot)
  fitDist(protStdevsInGroup, lambda x, shape, sigma: gamma.pdf(x, shape, 0.0, sigma), "stdev log10(protein diff in group)", ["shapeProteinStdevs", "scaleProteinStdevs"], params, plot, x = np.arange(-0.1, 1.0, 0.005))
  
  qc = params['proteinQuantCandidates']
  diffCandidates = np.linspace(2*qc[0], 2*qc[-1], len(qc)*2-1)
  params['priorDiffsVarSigma'] = funcHypsec(diffCandidates, params['muProteinDiffs'], params['sigmaCandidates'][:, np.newaxis])
  #fitDist(protGroupDiffs, funcHypsec, "log10(protein diff between groups)", ["muProteinGroupDiffs", "sigmaProteinGroupDiffs"], params, plot)

# this heuristic does not report a protein missing if 0/1, 1/2, 2/4 are observed, 
# but will for example report 1/3 and 3/6 observed as missing
def proteinMissingHeuristic2(numObs, numRows):
  return (float(numObs) + 1) / (numRows) < 0.7

def proteinMissingHeuristic(numObs, numRows, observedProbPerRun):
  #if proteinMissingHeuristic2(numObs, numRows) and binom.cdf(numObs+1, numRows+1, observedProbPerRun) >= 0.1:
  #  print(numObs, numRows, observedProbPerRun, binom.cdf(numObs+1, numRows+1, observedProbPerRun))
  return binom.cdf(numObs, numRows, observedProbPerRun) < 0.1
  
def imputeMLEValue(row, numObs, mleXIC, numRows, observedProbPerRun):
  #if proteinMissingHeuristic2(numObs, numRows):
  if proteinMissingHeuristic(numObs, numRows, observedProbPerRun):
    return mleXIC
  elif numObs == 0:
    return np.nan
  else:
    return parsers.geomAvg(row)
  
def fitLogitNormal(observedValues, params, plot):
  m = np.mean(observedValues)
  s = np.std(observedValues)
  minBin, maxBin = m - 4*s, m + 4*s
  #minBin, maxBin = -2, 6
  if plot:
    plt.figure()
    (vals, bins, patches) = plt.hist(observedValues, bins = np.arange(minBin, maxBin, 0.1), normed = True, alpha = 0.5, label = "observed intensities")
  else:
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
    plt.plot(bins, norm.pdf(bins, np.mean(observedValues), np.std(observedValues)), label='normal fit')
    plt.plot(bins, funcLogitNormal(bins, *popt), label='logit-normal fit')
    plt.plot(bins, 0.5 + 0.5 * np.tanh((np.array(bins) - popt[0]) / popt[1]), label = "detection ratio")
    plt.plot(bins, norm.pdf(bins, popt[2], popt[3]), label = "underlying distribution")
    plt.xlabel("log10(intensity)")
    plt.legend()

def updateMissingLogit(observedXICValues, imputedVals, params, plot):
  if plot:
    plt.figure()
    plt.subplot(2,1,1)
  minX = min([-2, params["muXIC"] - 4*params["sigmaXIC"]])
  maxX = min([6, params["muXIC"] + 4*params["sigmaXIC"]])
  bins = np.arange(minX, maxX, 0.1)
  #bins = np.arange(-2, 6, 0.1)
  (vals1, _, _) = plt.hist(observedXICValues, bins = bins, alpha = 0.5, label = 'observed')
  (vals2, _, _) = plt.hist(imputedVals, bins = bins, alpha = 0.5, label = 'imputed')
  
  x = bins[:-1]
  observedPdf = funcLogitNormal(x, params["muDetect"], params["sigmaDetect"], params["muXIC"], params["sigmaXIC"]) / 10
  totalPdf = funcNorm(x, params["muXIC"], params["sigmaXIC"]) / 10
  numTotal = len(observedXICValues) / np.sum(observedPdf)
  missing = np.max([np.zeros_like(x), totalPdf * numTotal - np.max([observedPdf * numTotal, vals1], axis = 0) - vals2], axis = 0)
  missing2 = np.max([np.zeros_like(x), totalPdf * numTotal - np.max([observedPdf * numTotal, vals1], axis = 0)], axis = 0)
  
  totalPdfUpdated = (vals1 + vals2 + missing + 1e-10)
  y = vals1 / totalPdfUpdated
  popt, pcov = curve_fit(logit, x, y) #, sigma = 1.0 / np.log10(vals1 + 1))
  if plot:
    plt.plot(x, missing2)
    plt.plot(x, missing + vals2)
    
    plt.subplot(2,1,2)
    plt.plot(x, y, 'b')
    plt.plot(x, vals2 / (vals1 + vals2), 'm')
    plt.plot(x, logit(x, *popt), 'r')
    plt.plot(x, logit(x, params["muDetect"], params["sigmaDetect"]), 'g')
  params["muDetect"], params["sigmaDetect"] = popt[0], popt[1]
  print("params[\"muDetect\"], params[\"sigmaDetect\"] = %f, %f" % (popt[0], popt[1]))
    
def fitDist(ys, func, xlabel, varNames, params, plot, x = np.arange(-2,2,0.01)):
  if plot:
    plt.figure()
    (vals, bins, patches) = plt.hist(ys, bins = x, normed = True)
  else:
    vals, bins = np.histogram(ys, bins = x, normed = True)
  bins = bins[:-1]
  popt, pcov = curve_fit(func, bins, vals)
  outputString = ", ".join(["params[\"%s\"]"]*len(popt)) + " = " + ", ".join(["%f"] * len(popt))
  for varName, val in zip(varNames, popt):
    params[varName] = val
  print(outputString % tuple(varNames + list(popt)))
  if plot:
    plt.plot(bins, func(bins, *popt), label='distribution fit')
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

if __name__ == "__main__":
  main(sys.argv[1:])
