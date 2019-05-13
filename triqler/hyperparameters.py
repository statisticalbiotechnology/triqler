#!/usr/bin/python

from __future__ import print_function

import csv
import sys
import os
import itertools

import numpy as np
from scipy.stats import hypsecant, gamma, norm, binom, expon #, cauchy
from scipy.optimize import curve_fit

from . import parsers

def fitPriors(peptQuantRows, params, printImputedVals = False, plot = False):
  ########
  # ToDo #
  # - FIX params["groupNorm"] = True, remove redundancy of geoAvgQuantRowGroup !!!!!!!!
  ########
    
  params['proteinQuantCandidates'] = np.arange(-5.0, 5.0 + 1e-10, 0.01) # log10 of protein ratio  
  qc = params['proteinQuantCandidates']
  params['proteinDiffCandidates'] = np.linspace(2*qc[0], 2*qc[-1], len(qc)*2-1)
  
  #print(params["groups"])
  #print(params["groupLabels"])
  

  protQuantRows = parsers.filterAndGroupPeptides(peptQuantRows, lambda x : not x.protein[0].startswith(params['decoyPattern']))

  imputedVals, imputedDiffs, observedXICValues, protQuants, protDiffs, protStdevsInGroup, protGroupDiffs = list(), list(), list(), list(), list(), list(), list()
  observedXICValuesGroups = [list() for i in range(len(params["groupLabels"]))] # observedXICValuesGroups for each condition/group.
  protQuantsGroups = [list() for i in range(len(params["groupLabels"]))] # ProtQuant for each condition/group.
  quantRowsCollection = list()
  quantRowsCollectionGroup = list() # for group-wise samples.
  count = 0
  imputedDiffsGroup = [[] for i in range(len(params["groupLabels"]))] # for imputedDiffsGroup_i
      
  for prot, quantRows in protQuantRows:
    
    quantRows, quantMatrix = parsers.getQuantMatrix(quantRows)
    
    # GROUP BASED NORMALIZATION AND AVERAGING
    #print(params["groupNorm"])
    if params["groupNorm"] == True:
        quantMatrixNormalizedGroup = []
        for row in quantMatrix:
            quantMatrixNormalized_i = []
            for i in range(len(params["groupLabels"])):
                quantMatrixNormalized_i.append(parsers.geoNormalize(np.array(row)[params["groups"][i]]))
            quantMatrixNormalized_i = np.concatenate(quantMatrixNormalized_i)
            quantMatrixNormalizedGroup.append(quantMatrixNormalized_i)
        quantRowsCollectionGroup.append((quantRows, quantMatrixNormalizedGroup))
        geoAvgQuantRowGroup = getProteinQuant(quantMatrixNormalizedGroup, quantRows)
        geoAvgQuantRow = geoAvgQuantRowGroup
    else:
    # NORMALIZE BASED ON ALL DATA
        quantMatrixNormalized = [parsers.geoNormalize(row) for row in quantMatrix]
        quantRowsCollection.append((quantRows, quantMatrix))
        geoAvgQuantRow = getProteinQuant(quantMatrixNormalized, quantRows)
        
    #print(np.array(quantMatrixNormalized)[:,0:3])
    
    #print(getProteinQuant(quantMatrixNormalized[0:3], quantRows))
    #print(geoAvgQuantRow[params["groups"][0]]) 
    #print(geoAvgQuantRow) #is np.array
    #if params["knownGroups"] == True:
      #params["muProteinGroups"] = ['muProteinGroup%s' % s for s in params["groupLabels"]] # CREATE DIFFERENT mu FOR GROUPS
      #params["sigmaProteinGroups"] = ['sigmaProteinGroup%s' % s for s in params["groupLabels"]] # CREATE sigma FOR GROUPS
      #for i in range(len(params["groupLabels"])):
      #    print(params["muProteinGroups"][i])
      #    print(params["sigmaProteinGroups"][i])
      
    #print(params["groups"])
    if params["knownGroups"] == True:
        if params["groupNorm"] == True:
            for i in range(len(params["groupLabels"])):
                protQuantsGroups[i].extend([np.log10(x) for x in geoAvgQuantRowGroup[params["groups"][i]] if not np.isnan(x)])    
        else:
            for i in range(len(params["groupLabels"])):
                protQuantsGroups[i].extend([np.log10(x) for x in geoAvgQuantRow[params["groups"][i]] if not np.isnan(x)])    
    protQuants.extend([np.log10(x) for x in geoAvgQuantRow if not np.isnan(x)])
    #print(len(protQuants))
    #print(protQuants)
    
    if params["groupNorm"] == True:
        args = parsers.getQuantGroups(geoAvgQuantRowGroup, params["groups"], np.log10)
    else:
        args = parsers.getQuantGroups(geoAvgQuantRow, params["groups"], np.log10)

    #means = list()
    
    for group in args: # ALREADY SPLIT UP INTO GROUPS
      if np.count_nonzero(~np.isnan(group)) > 1:
        protDiffs.extend(group - np.mean(group))
        protStdevsInGroup.append(np.std(group))
      #if np.count_nonzero(~np.isnan(group)) > 0:
      #  means.append(np.mean(group))
    
    #if np.count_nonzero(~np.isnan(means)) > 1:
    #  for mean in means:  
    #    #protGroupDiffs.append(mean - np.mean(means))
    #    protGroupDiffs.append(mean)
    
    quantMatrixFiltered = np.log10(np.array([x for x, y in zip(quantMatrix, quantRows) if y.combinedPEP < 1.0]))  
    
    #[print(len(x)) for x, y in zip(quantMatrix, quantRows) if y.combinedPEP < 1.0]
    #print(np.shape(np.array([x for x, y in zip(quantMatrix, quantRows) if y.combinedPEP < 1.0])))
    
    # GET observedXICValuesGroups for each group. Is this correct???
    if params["knownGroups"] == True:
        #smurfVals = quantMatrixFiltered[~np.isnan(quantMatrixFiltered)]
        #smurfComp = []
        for i in range(len(params["groupLabels"])):
            for j in range(len(quantMatrixFiltered)):
            #print(smurfVals)
                observedXICValuesGroups[i].extend(quantMatrixFiltered[j][params["groups"][i]][~np.isnan(quantMatrixFiltered[j][params["groups"][i]])])
                #smurfComp.extend(quantMatrixFiltered[i][params["groups"][i]][~np.isnan(quantMatrixFiltered[i][params["groups"][i]])])
        #print(len(smurfVals))
        #print(len(smurfComp))   
        #if len(smurfVals) != len(smurfComp):
        #    print(smurfVals)
        #    print(smurfComp)
    #print(quantMatrixFiltered)
    #print(type((quantMatrixFiltered)))
    #print(type((quantMatrixFiltered[0])))
    #print(len(quantMatrixFiltered))
    #print(len(quantMatrixFiltered[0]))
    #print(quantMatrixFiltered[~np.isnan(quantMatrixFiltered)])
    #print(quantMatrixFiltered[0][~np.isnan(quantMatrixFiltered[0])])
  #  print(quantMatrixFiltered)
 #   print(quantMatrixFiltered[~np.isnan(quantMatrixFiltered)])
    #print("BREAKPOINT")
    #if params["knownGroups"] == True:
    #    for i in range(len(params["groupLabels"])):
    #        observedXICValuesGroups[i].extend
    
    observedXICValues.extend(quantMatrixFiltered[~np.isnan(quantMatrixFiltered)])
    
    #print(len(observedXICValuesGroups[0]))
    #print(len(observedXICValuesGroups[1]))
    #print(len(observedXICValuesGroups[2]))
    #print(len(observedXICValues))
    
    #print(observedXICValues)
    #print((len(observedXICValuesGroups[0])+len(observedXICValuesGroups[1])+len(observedXICValuesGroups[2])) == (len(observedXICValues)))
    # counts number of NaNs per run, if there is only 1 non NaN in the column, we cannot use it for estimating the imputedDiffs distribution
    
    #print(quantMatrixFiltered[:, params["groups"][0]])
#    
#    numNonNaNsGroups = []
#    for i in range(len(quantMatrixFiltered)):
#        numNonNaNs_i = []
#        for j in range(len(params["groupLabels"])):
#            numNonNaNs_i.append(np.count_nonzero(~np.isnan(quantMatrixFiltered[i, params["groups"][i]]), axis = 0))
#        print(numNonNaNs_i)
#        
#    for i in range(len(params["groupLabels"])):
#        numNonNaNsGroups.extend(np.count_nonzero(~np.isnan(quantMatrixFiltered[:, params["groups"][i]]), axis = 0)[np.newaxis,:])
#    
#    numNonNaNsGroups = np.concatenate(numNonNaNsGroups)

    numNonNaNs = np.count_nonzero(~np.isnan(quantMatrixFiltered), axis = 0)[np.newaxis,:]

    # Edit to make imputed diffs for each groups
    if params["groupNorm"] == True:
        xImpsGroups = []
        for i in range(len(params["groupLabels"])):
            xImps_i = imputeValues(quantMatrixFiltered[:, params["groups"][i]], 
                                   geoAvgQuantRow[params["groups"][i]], 
                                   np.log10(geoAvgQuantRow[params["groups"][i]]))
            xImpsGroups.append(xImps_i)
        xImpsGroups = np.concatenate(xImpsGroups, axis = 1)
        xImps = xImpsGroups
        
        # Different feature impute for each group! <---------
        for i in range(len(params["groupLabels"])):
        #    print(xImps[:, params["groups"][i]] - )
            imputedDiffsGroup[i].extend((xImps[:, params["groups"][i]] - quantMatrixFiltered[:,params["groups"][i]])[(~np.isnan(quantMatrixFiltered[:,params["groups"][i]])) & (np.array(numNonNaNs[0][params["groups"][i]]) > 1)])
        
        # Global features
        imputedDiffs.extend((xImps - quantMatrixFiltered)[(~np.isnan(quantMatrixFiltered)) & (np.array(numNonNaNs) > 1)])
    else:
        xImps = imputeValues(quantMatrixFiltered, geoAvgQuantRow, np.log10(geoAvgQuantRow))
        imputedDiffs.extend((xImps - quantMatrixFiltered)[(~np.isnan(quantMatrixFiltered)) & (np.array(numNonNaNs) > 1)])
    
    #print(np.shape(xImpsGroups) == np.shape(xImps))
    #print(xImpsGroups == xImps)
    
    
    #imputedVals.extend(xImps[(np.isnan(quantMatrixFiltered)) & (np.array(numNonNaNs) > 1)])
  if params["knownGroups"] == True:
      for i in range(len(observedXICValuesGroups)):
          fitLogitNormal(observedXICValuesGroups[i], ["muDetect"+params["groupLabels"][i],
                                                 "sigmaDetect"+params["groupLabels"][i],
                                                 "muXIC"+params["groupLabels"][i],
                                                 "sigmaXIC"+params["groupLabels"][i]],params, plot) 
  fitLogitNormal(observedXICValues, ["muDetect", "sigmaDetect", "muXIC", "sigmaXIC"], params, plot) # NOW I AM HERE 2019-04-17
  #print(params.keys())
  #print(len(protQuants))
  #print(protQuants)
  #print(params.keys())
  #print(params["fileList"])
  if params["knownGroups"] == True:
      for i in range(len(protQuantsGroups)):  
          fitDist(protQuantsGroups[i], funcHypsec, "log10(protein ratio) group"+params["groupLabels"][i],
                  ["muProteinGroup"+params["groupLabels"][i], "sigmaProteinGroup"+params["groupLabels"][i]],
                  params, plot)
      #print(params["groupLabels"][i])
      #print(params["muProteinGroup"+params["groupLabels"][i]])
      #print(params["sigmaProteinGroup"+params["groupLabels"][i]])
  fitDist(protQuants, funcHypsec, "log10(protein ratio)", ["muProtein", "sigmaProtein"], params, plot)#, THATAG = True)
  #fitDist(protQuants, funcExpon, "log10(protein ratio)", ["muProtein", "sigmaProtein"], params, plot)#, THATAG = True)
  #print("GLOBAL")
  #print(params["muProtein"])
  #print(params["sigmaProtein"])
  
  #print(params["muDetect"])
  #print(params["muProtein"])
  
  #################################################################
  # CHANGE THIS TO INCORPORATE MULTIPLE muFeatureDiff? 2019-04-29 #
  #################################################################
  #print(len(imputedDiffs))
  fitDist(imputedDiffs, funcHypsec, "log10(imputed xic / observed xic)", ["muFeatureDiff", "sigmaFeatureDiff"], params, plot)
  
  if params["knownGroups"] == True:
      for i in range(len(imputedDiffsGroup)):
          try: # #Try yo find 
              fitDist(imputedDiffsGroup[i], funcHypsec, "log10(imputed xic / observed xic)" + params["groupLabels"][i],
                      ["muFeatureDiff"+params["groupLabels"][i], "sigmaFeatureDiff"+params["groupLabels"][i]],
                      params, plot)
          except: # IS THIS OK??? <---------------------------- CHECK THIS
              print("WARNING! Optimal parameters not found for group " + params["groupLabels"][i] +", using global features instead for " + params["groupLabels"][i] + "!")
              fitDist(imputedDiffs, funcHypsec, "log10(imputed xic / observed xic)" + params["groupLabels"][i],
                      ["muFeatureDiff"+params["groupLabels"][i], "sigmaFeatureDiff"+params["groupLabels"][i]],
                      params, plot)
  
  fitDist(protStdevsInGroup, funcGamma, "stdev log10(protein diff in group)", ["shapeInGroupStdevs", "scaleInGroupStdevs"], params, plot, x = np.arange(-0.1, 1.0, 0.005))
  #print(len(protStdevsInGroup))
  sigmaCandidates = np.arange(0.001, 3.0, 0.001)
  gammaCandidates = funcGamma(sigmaCandidates, params["shapeInGroupStdevs"], params["scaleInGroupStdevs"])
  support = np.where(gammaCandidates > max(gammaCandidates) * 0.01)
  params['sigmaCandidates'] = np.linspace(sigmaCandidates[support[0][0]], sigmaCandidates[support[0][-1]], 20)
  #print(params)
  #params['proteinPrior'] = funcLogHypsec(params['proteinQuantCandidates'], params["muProtein"], params["sigmaProtein"]) ### HERE IS THE PRIOR for PROTEIN!
  #print(params["groups"])
  #print(params["groupLabels"])
  if params["knownGroups"] == True:
      params["proteinPriorGroups"] = ['proteinPriorGroup%s' % s for s in params["groupLabels"]] # CREATE DIFFERENT PRIORS FOR GROUPS
      for i in range(len(params["proteinPriorGroups"])):
          params[params["proteinPriorGroups"][i]] = funcLogHypsec(params['proteinQuantCandidates'], 
                params["muProteinGroup"+params["groupLabels"][i]], 
                params["sigmaProteinGroup"+params["groupLabels"][i]])
          #params[i] = funcLogHypsec(params['proteinQuantCandidates'], params["muProtein"], params["sigmaProtein"])
      #print(params.keys())
      #print(params["proteinPriorGroups"])
  #print(params['proteinQuantCandidates'])
  params['proteinPrior'] = funcLogHypsec(params['proteinQuantCandidates'], params["muProtein"], params["sigmaProtein"]) ### HERE IS THE PRIOR for PROTEIN!
  #print(params["proteinPriorGroups"][0])
  #print(params[params["proteinPriorGroups"][0]])
  #print(len(params[params["proteinPriorGroups"][0]]))
  #print(params["proteinPriorGroups"][1])
  #print(params[params["proteinPriorGroups"][1]])
  #print(len(params[params["proteinPriorGroups"][1]]))
  #print(params["proteinPriorGroups"][2])
  #print(params[params["proteinPriorGroups"][2]])
  #print(len(params[params["proteinPriorGroups"][2]]))
  #print("GLOBAL")
  #print(params["proteinPrior"])
  #print(len(params["proteinPrior"]))
  #ToDo
  #NEED TO FIND DIFFERENT muProteins and sigmaProteins for different groups
  
  #params['proteinPrior'] = funcExpon(params['proteinQuantCandidates'], loc = -5000, shape = 100)
  #print(params['proteinPrior'])
  #np.savetxt("smurf.csv", params['proteinPrior'], delimiter = "\t")
  #import pandas as pd
  #smurfDF = pd.DataFrame(params["proteinPrior"])
  #smurfax = smurfDF.plot()
  #print(smurfDF.sum())
  #smurfFig = smurfax.get_figure()
  #smurfFig.savefig("smurffig.png")
  #print("PRINTING POSTERIOR")
  #print(params['proteinPrior'])
  #print(len(params['proteinPrior']))
  

  # IDENTIFY WHICH VALUE ACTUALLY BECOMES THE PROTEIN QUANTIFICATION...
  if "shapeInGroupStdevs" in params:
    params['inGroupDiffPrior'] = funcHypsec(params['proteinDiffCandidates'], 0, params['sigmaCandidates'][:, np.newaxis])
  else: # if we have technical replicates, we could use a delta function for the group scaling parameter to speed things up
    fitDist(protDiffs, funcHypsec, "log10(protein diff in group)", ["muInGroupDiffs", "sigmaInGroupDiffs"], params, plot)
    params['inGroupDiffPrior'] = funcHypsec(params['proteinDiffCandidates'], params['muInGroupDiffs'], params['sigmaInGroupDiffs'])
  #fitDist(protGroupDiffs, funcHypsec, "log10(protein diff between groups)", ["muProteinGroupDiffs", "sigmaProteinGroupDiffs"], params, plot)
  
def fitLogitNormal(observedValues, varNames, params, plot):
  m = np.mean(observedValues)
  s = np.std(observedValues)
  minBin, maxBin = m - 4*s, m + 4*s
  #minBin, maxBin = -2, 6
  vals, bins = np.histogram(observedValues, bins = np.arange(minBin, maxBin, 0.1), normed = True)
  bins = bins[:-1]
  popt, _ = curve_fit(funcLogitNormal, bins, vals, p0 = (m, s, m - s, s))
  
  varNames = varNames
  for varName, val in zip(varNames, popt):
      #print(varName + " " + str(val))
      params[varName] = val
  # WHAT THE FUCK IS muDetect and does the PGM Go both ways?????
  
  #print("params[\"muDetectInit\"], params[\"sigmaDetectInit\"] = %f, %f" % (popt[0], popt[1]))
  print("params[\""+varNames[0]+"\"], params[\""+varNames[1]+"\"] = %f, %f" % (popt[0], popt[1]))
  print("params[\""+varNames[2]+"\"], params[\""+varNames[3]+"\"] = %f, %f" % (popt[2], popt[3]))
  #params["muDetectInit"], params["sigmaDetectInit"] = popt[0], popt[1]
  #params["muDetect"], params["sigmaDetect"] = popt[0], popt[1]
  #params["muXIC"], params["sigmaXIC"] = popt[2], popt[3]
  if plot:
    poptNormal, _ = curve_fit(funcNorm, bins, vals)
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.bar(bins, vals, width = bins[1] - bins[0], alpha = 0.5, label = 'observed distribution')
    plt.plot(bins, funcLogitNormal(bins, *popt), 'g', label='logit-normal fit', linewidth = 2.0)
    plt.plot(bins, 0.5 + 0.5 * np.tanh((np.array(bins) - popt[0]) / popt[1]), 'm', label = "logit-part fit", linewidth = 2.0)
    plt.plot(bins, funcNorm(bins, popt[2], popt[3]), 'c', label = "normal-part fit", linewidth = 2.0)
    plt.plot(bins, funcNorm(bins, *poptNormal), 'r', label='normal fit', linewidth = 2.0)
    plt.xlabel("log10(intensity)", fontsize = 18)
    plt.legend()
    
def fitDist(ys, func, xlabel, varNames, params, plot, x = np.arange(-2,2,0.01)):
  vals, bins = np.histogram(ys, bins = x, normed = True)
  bins = bins[:-1]
  popt, _ = curve_fit(func, bins, vals)
  outputString = ", ".join(["params[\"%s\"]"]*len(popt)) + " = " + ", ".join(["%f"] * len(popt))
  for varName, val in zip(varNames, popt):
    params[varName] = val
  
  if func == funcHypsec:
    fitLabel = "hypsec fit"
  elif func == funcNorm:
    fitLabel = "normal fit"
  elif func == funcGamma:
    fitLabel = "gamma fit"
  else:
    fitLabel = "distribution fit"
  print(outputString % tuple(varNames + list(popt)))
  if plot:    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.bar(bins, vals, width = bins[1] - bins[0], label = 'observed distribution')
    plt.plot(bins, func(bins, *popt), 'g', label=fitLabel, linewidth = 2.0)
    if func == funcHypsec:
      poptNormal, _ = curve_fit(funcNorm, bins, vals)
      plt.plot(bins, funcNorm(bins, *poptNormal), 'r', label = 'normal fit', linewidth = 2.0)
      
      if True:
        funcStudentT = lambda x, df, mu, sigma : t.pdf(x, df = df, loc = mu, scale = sigma)
        poptStudentT, _ = curve_fit(funcStudentT, bins, vals)
        print(poptStudentT)
        
        funcCauchy = lambda x, mu, sigma : cauchy.pdf(x, loc = mu, scale = sigma)
        poptCauchy, _ = curve_fit(funcCauchy, bins, vals)
        print(poptCauchy)
        
        plt.plot(bins, funcStudentT(bins, *poptStudentT), 'm', label = 'student-t fit', linewidth = 2.0)
        plt.plot(bins, funcCauchy(bins, *poptCauchy), 'c', label = 'cauchy fit', linewidth = 2.0)
        
        funcLogStudentT = lambda x, df, mu, sigma : t.logpdf(x, df = df, loc = mu, scale = sigma)
        funcLogNorm = lambda x, mu, sigma : norm.logpdf(x, loc = mu, scale = sigma)
        funcLogCauchy = lambda x, mu, sigma : cauchy.logpdf(x, loc = mu, scale = sigma)
        
        plt.xlabel(xlabel, fontsize = 18)
        plt.legend()
        
        plt.figure()
        plt.plot(bins, funcLogHypsec(bins, *popt), 'g', label = 'hypsec log fit', linewidth = 2.0)
        plt.plot(bins, funcLogNorm(bins, *poptNormal), 'r', label = 'normal log fit', linewidth = 2.0)
        plt.plot(bins, funcLogStudentT(bins, *poptStudentT), 'm', label = 'student-t log fit', linewidth = 2.0)
        plt.plot(bins, funcLogCauchy(bins, *poptCauchy), 'c', label = 'cauchy log fit', linewidth = 2.0)
    plt.xlabel(xlabel, fontsize = 18)
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
  #return cauchy.pdf(x, mu, sigma)
  #return norm.pdf(x, mu, sigma)

def funcLogHypsec(x, mu, sigma):
  return hypsecant.logpdf(x, mu, sigma)
  #return cauchy.logpdf(x, mu, sigma)
  #return norm.logpdf(x, mu, sigma)
  
def funcGamma(x, shape, sigma):
  return gamma.pdf(x, shape, 0.0, sigma)
  
def logit(x, muLogit, sigmaLogit):
  return 0.5 + 0.5 * np.tanh((np.array(x) - muLogit) / sigmaLogit)

def funcExpon(x, loc = 0, shape = 1):
  return expon.logpdf(x, loc, shape)
    
