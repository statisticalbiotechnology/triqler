#!/usr/bin/python

from __future__ import print_function

import itertools

import numpy as np
from scipy.stats import f_oneway, gamma
from scipy.optimize import curve_fit

from . import parsers
from . import convolution_dp
from . import hyperparameters

import sys

def getPosteriors(quantRowsOrig, params, returnDistributions = True):
  
  #print(params)
  quantRows, quantMatrix = parsers.getQuantMatrix(quantRowsOrig)
  
  pProteinQuantsList, bayesQuantRow = getPosteriorProteinRatios(quantMatrix, quantRows, params)
  pProteinGroupQuants = getPosteriorProteinGroupRatios(pProteinQuantsList, bayesQuantRow, params)
  pProteinGroupDiffs, muGroupDiffs = getProteinGroupsDiffPosteriors(pProteinGroupQuants, params)
  
  probsBelowFoldChange = getProbBelowFoldChangeDict(pProteinGroupDiffs, params)
  if returnDistributions:
    #print("returnDistributions is TRUE!")
    #sys.stdout.flush()
    return bayesQuantRow, muGroupDiffs, probsBelowFoldChange, pProteinQuantsList, pProteinGroupQuants, pProteinGroupDiffs
  else:
    #print("returnDistributions is FALSE!")
    #sys.stdout.flush()
    return bayesQuantRow, muGroupDiffs, probsBelowFoldChange

def getPosteriorProteinRatios(quantMatrix, quantRows, params, maxIterations = 50, bayesQuantRow = None):
  #print(len(quantMatrix))
  #print(params.keys())
  #print("")
  numSamples = len(quantMatrix[0])
  #print(numSamples)
  bayesQuantRow = np.array([1.0]*numSamples) #<--- uniform prior????
  
  for iteration in range(maxIterations):  
    prevBayesQuantRow = np.copy(bayesQuantRow)
    pProteinQuantsList, bayesQuantRow = getPosteriorProteinRatio(quantMatrix, quantRows, bayesQuantRow, params)
    #print(len(pProteinQuantsList[8]))    
    #print(len(pProteinQuantsList[0]))
    
    #print(bayesQuantRow)
    bayesQuantRow = parsers.geoNormalize(bayesQuantRow)
    
    diffInIteration = np.log10(prevBayesQuantRow) - np.log10(bayesQuantRow)
    if np.max(diffInIteration*diffInIteration) < 1e-4:
      #print("Converged after iteration", iteration+1)
      break
  
  return pProteinQuantsList, bayesQuantRow

def getPosteriorProteinRatio(quantMatrix, quantRows, geoAvgQuantRow, params):
  numSamples = len(quantMatrix[0])
  
  #for row in quantMatrix:
  #    print(row)
  #    print("NEXTROW")
  
  
  #logGeoAvgsGroups = [list() for i in range(len(params["groupLabels"]))]
  #featDiffsGroups = [list() for i in range(len(params["groupLabels"]))] #<------------- check this!
  #pMissingGeomAvgGroups = [list() for i in range(len(params["groupLabels"]))]
  
  
  #for row in quantMatrix:
  #    for i in range(len(params["groupLabels"])):
          #print(row[params["groups"][i]])
          #print(row[params["groups"][i]])
          #print(parsers.geomAvg(row[params["groups"][i]]))
  #        logGeoAvgsGroups[i] = np.log10(parsers.geomAvg(row[params["groups"][i]]))
  #        print(parsers.geomAvg(row[params["groups"][i]]))
  #        featDiffsGroups[i] = np.log10(np.array(quantMatrix)[:,params["groups"][i]]) - logGeoAvgsGroups[i]
  #        pMissingGeomAvgGroups[i] = pMissing(logGeoAvgsGroups[i], params["muDetect"+params["groupLabels"][i]],
  #                             params["sigmaDetect"+params["groupLabels"][i]])
  #    print("#############################################")
  
  # ABOVE IS HARD TO SEE WHAT IS GOING ON REDO IN MULTIPLE FOR-LOOPS
  #for row in quantMatrix:
  #    print(parsers.geomAvg(row))
  #[print(parsers.geomAvg(row)) for row in quantMatrix]
  
  #print(np.shape(quantMatrix))
  

  logGeoAvgsGroups = []
  for row in quantMatrix:
      logGeoAvgs_i = []
      for i in range(len(params["groupLabels"])):
          logGeoAvgs_i.append(np.log10(parsers.geomAvg(row[params["groups"][i]])))
      logGeoAvgsGroups.append(logGeoAvgs_i)

  
  # featDiffs should not need to be split, since they are quantMatrix - logGeoAvgsGroups
  #for i in logGeoAvgsGroups:
  #  for j in range(len(params["groupLabels"])):
  
  ########
  # MATC logGEoAVGS to quantMatrix!!!! ###
          ###

  featDiffsGroups = []
  for row in range(len(quantMatrix)):
      featDiffs_i = []
      for i in range(len(params["groupLabels"])):
          #print("qMAT")
          #print(np.log10(quantMatrix[row][params["groups"][i]]))
          #print("logGeo")
          #print(logGeoAvgsGroups[row][i])
          #print(logGeoAvgsGroups[row][params["groups"][i]])
          featDiffs_i.append(np.log10(quantMatrix[row][params["groups"][i]]) - logGeoAvgsGroups[row][i])
      featDiffsGroups.append(np.concatenate(featDiffs_i, axis = 0))#.tolist())
  featDiffsGroups = np.array(featDiffsGroups)  
  #print(featDiffsGroups)
  #print("NEWLINE###################")   # I THINK THIS IS DONE!

  pMissingGeomAvgGroups = [[] for i in range(len(params["groupLabels"]))]
  #print(pMissingGeomAvgGroups)
  #print(logGeoAvgsGroups) 
#  print(logGeoAvgs)
  for i in range(len(params["groupLabels"])):
      #print(np.array(logGeoAvgsGroups)[:,i])
      #print(pMissing(np.array(logGeoAvgsGroups)[:,i], 
      #      params["muDetect"+params["groupLabels"][i]], 
      #      params["sigmaDetect"+params["groupLabels"][i]]))
      pMissingGeomAvgGroups[i] = pMissing(np.array(logGeoAvgsGroups)[:,i], 
            params["muDetect"+params["groupLabels"][i]], 
            params["sigmaDetect"+params["groupLabels"][i]])
  pMissingGeomAvgGroups = np.array(pMissingGeomAvgGroups).T.tolist()

  ############################################################
  # DONE WITH  pMissingGeomAvg Prior Parameters 2019-04-29 ###
  ############################################################ 

  logGeoAvgs = np.log10([parsers.geomAvg(row) for row in quantMatrix])
  featDiffs = np.log10(quantMatrix) - logGeoAvgs[:,np.newaxis] 
  
  featDiffs = featDiffsGroups #<---------------------------------------- NOTE THIS!!!!
  pMissingGeomAvg = pMissing(logGeoAvgs, params["muDetect"], params["sigmaDetect"]) # Pr(f_grn = NaN | t_grn = 1)
  
  
  #print(logGeoAvgsGroups) 
  #print(params[

  #print(featDiffs == featDiffsGroups)
  
  #for i in range(len(logGeoAvgsGroups)):
  #    print
#  for i in range(len(params["groupLabels"])):
#      print(params["sigmaDetect"+params["groupLabels"][i]])
  
  #for i in logGeoAvgsGroups:
      
  #print(np.shape(featDiffs) == np.shape(featDiffsGroups))
  #  print(quantMatrix)
#  print("qMatrix")   
#  print(np.log10(quantMatrix))
#  print("logGeos")
#  print(logGeoAvgs[:,np.newaxis])    
#  print("###############")
#  print("NEEEXT")
  #print(len(featDiffs[0]))
  #print(len(featDiffs) == len(logGeoAvgs))
  '''
  print("""
        CHECK params!!!!!
        params["muDetect"] and params["sigmaDetect"]
        """)
  print(params["muDetect"])
  print(params["sigmaDetect"])
  print("""
        ###############
        END OF CHECKING
        ###############
        """)
  #print(pMissingGeomAvg)
  #print(featDiffsGroups)
  #print("")
  '''
  #pQuantIncorrectId = hyperparameters.funcGamma(featDiffs, params["muFeatureDiff"], params["sigmaFeatureDiff"]) # Pr(f_grn = x | t_grn = 1)
  #Changed below
  #print(featDiffs == featDiffsGroups) # featDiffs is equal to featDiffsGroup!!!!
  #print(featDiffs)
  pQuantIncorrectId = hyperparameters.funcHypsec(featDiffs, params["muFeatureDiff"], params["sigmaFeatureDiff"]) # Pr(f_grn = x | t_grn = 1)
  
  pQuantIncorrectIdGroups = [[] for i in range(len(params["groupLabels"]))]
  for i in range(len(params["groupLabels"])):
      pQuantIncorrectIdGroups[i] = hyperparameters.funcHypsec(featDiffsGroups[:, params["groups"][i]], 
                                                          params["muFeatureDiff"+params["groupLabels"][i]], 
                                                          params["sigmaFeatureDiff"+params["groupLabels"][i]])
  pQuantIncorrectIdGroups = np.concatenate(pQuantIncorrectIdGroups, axis = 1)
  #print(np.shape(pQuantIncorrectIdGroups))
  #print(np.shape(pQuantIncorrectId))
  #print(np.shape(pQuantIncorrectId) == np.shape(pQuantIncorrectIdGroups))
  #pQuantIncorrectIdOld = hyperparameters.funcLogitNormal(np.log10(quantMatrix), params["muDetect"], params["sigmaDetect"], params["muXIC"], params["sigmaXIC"]) 
  
  #print(np.shape(quantMatrix))
  #print(np.shape(geoAvgQuantRow))
  
  xImpsAllGroups = [[] for i in range(len(params["groupLabels"]))]
  for i in range(len(params["groupLabels"])):
      #print(type(quantMatrix))
      #print(np.array(quantMatrix)[:, params["groups"][i]])
      #print(geoAvgQuantRow[params["groups"][i]])
      xImpsAllGroups[i] = imputeValues(np.array(quantMatrix)[:, params["groups"][i]],
                    geoAvgQuantRow[params["groups"][i]], params["proteinQuantCandidates"])
  xImpsAllGroups = np.concatenate(xImpsAllGroups, axis = 1)
  #print(np.shape(xImpsAllGroups))
  xImpsAll = imputeValues(quantMatrix, geoAvgQuantRow, params['proteinQuantCandidates'])
  #print(np.shape(xImpsAll))
  #print(np.shape(xImpsAll) == np.shape(xImpsAllGroups))
  #print((xImpsAll) == (xImpsAllGroups))
  impDiffs = xImpsAll - np.log10(np.array(quantMatrix))[:,:,np.newaxis]
  impDiffsGroups = xImpsAllGroups - np.log10(np.array(quantMatrix))[:,:,np.newaxis]
  #pDiffs = hyperparameters.funcGamma(impDiffs, params["muFeatureDiff"], params["sigmaFeatureDiff"]) # Pr(f_grn = x | m_grn = 0, t_grn = 0)
  #Chenged below
  #print(len(impDiffs[0]))
  
  #print(np.shape(impDiffs))
  #print(np.shape(impDiffsGroups))
  #print(np.shape(impDiffs) == np.shape(impDiffsGroups))
  #print(np.shape(xImpsAll) == np.shape(xImpsAllGroups))
  #print(impDiffs)
  #print(impDiffsGroups)
  
  pDiffs = hyperparameters.funcHypsec(impDiffs, params["muFeatureDiff"], params["sigmaFeatureDiff"]) # Pr(f_grn = x | m_grn = 0, t_grn = 0)
  #print(np.shape(pDiffs))
  
  pDiffsGroups = [[] for i in range(len(params["groupLabels"]))]
  for i in range(len(params["groupLabels"])):
      pDiffsGroups[i] = hyperparameters.funcHypsec(impDiffsGroups[:, params["groups"][i],:],
                  params["muFeatureDiff" + params["groupLabels"][i]],
                  params["sigmaFeatureDiff" + params["groupLabels"][i]])
  pDiffsGroups = np.concatenate(pDiffsGroups, axis = 1)
  #print(np.shape(pDiffsGroups) == np.shape(pDiffs))
  
  #print(params["muFeatureDiff"]) #<--------------
  
  
  #for i in range(len(params["groupLabels"])):
  #    featDiffs[:,params["groups"][0]]) 
  
  ########################################
  # CHANGING ABOVE PARAMETERS 2019-04-29 #
  ########################################
  
  #print(np.shape(featDiffsGroups) == np.shape(featDiffs))
  #print(len(featDiffsGroups))
  
  ##################################################################
  # CONTINUE HERE AFTER FIXING muFeatureDiff in hyperparameters.py #
  ##################################################################
  
#  pQuantIncorrectIdGroup = []
#  for i in range(len(featDiffsGroups)):
#      pQuantIncorrectIdGroup_i = []
#      for j in range(len(params["groupLabels"])):
#          print(featDiffsGroups[i][params["groups"][j]], params["muFeatureDiff"+params["groupLab]])
#  
#  print(params)
#  
  #######################3
  
  #print(hyperparameters.funcHypsec(featDiffsGroups[0][0:3], params["muFeatureDiff"], params["sigmaFeatureDiff"]))
  
  # fix mapping function for params["groups"] so that we can use different priors for differen j in for-loop below.
  #val = 5
  #a =  [[1,2,3],[4,5,6],[7,8,9], [5,5,5]]
  #cnt = 0
  #for i,j in enumerate(params["groups"]):
  #for i,j in enumerate(a):
      #print(i,j)
  #    if val in j:
  #         if cnt > 0:
 #              raise ("COUNT +1 <----------------------------------------")
 #          print(params["proteinPriorGroups"][i])
 #          cnt += 1 # count to check if more than one prior.
  #val = 5 # j
  #for i,j in enumerate(params["groups"]):
  #    if val in j:
  #        print(params["proteinPriorGroups"][i])
          
          
  #print(params.keys())
  #print(np.shape(params["proteinPriorGroups"])) 
  
  #print(params["proteinPriorGroups"])
  #print(params[params["proteinPriorGroups"][1]])
  #print(params["proteinPrior"])
  #print(type(params["groups"]))
  pProteinQuantsList, bayesQuantRow = list(), list()
  for j in range(numSamples):
    #pProteinQuant = params['proteinPrior'].copy() # log likelihood <--------------------- not multiple priors
    #print(pProteinQuant)
    cnt = 0
    for priorGroup, sampleInPrior in enumerate(params["groups"]):
        if j in sampleInPrior:
            if cnt > 0:
                raise ("ERROR with multiple prior assignement <----------------------------------------")
            #print(params["proteinPriorGroups"])
            pProteinQuant = params[params["proteinPriorGroups"][priorGroup]].copy()
            #if np.isnan(pProteinQuant).sum() > 0:
                #print("NaN count: " + str(np.isnan(pProteinQuant).sum()))
                #print(pProteinQuant)
                #raise ("NaN encountered in groupPrior")
            cnt += 1  
            #################################################################################
            # 2019-04-30 START TO CHANGE pMissingGroup, featDiffGroup xImpAllGroup etc etc. #
            # I think params["muDetect" + params["groupLabels"][i]] exists
            ##################################################################################
    for i, row in enumerate(quantMatrix):
      for priorGroup, sampleInPrior in enumerate(params["groups"]):
          if j in sampleInPrior: #Will be one for each sample... [s01r01, s01r02,... s02r01, ... s10r05]
              """
              NOTE THIS CODE WILL MAKE IT UPDATE likelihood original times the number of groups,
              
              """
              linkPEP = quantRows[i].linkPEP[j]
              identPEP = quantRows[i].identificationPEP[j]
              if identPEP < 1.0:
                pMissings = pMissing(xImpsAll[i,j,:],
                                     params["muDetect"+params["groupLabels"][priorGroup]],
                                     params["sigmaDetect"+params["groupLabels"][priorGroup]]) # Pr(f_grn = NaN | m_grn = 1, t_grn = 0)
                #pMissings_pseudo = pMissing(xImpsAllGroups[i,j,:], params["muDetect"], params["sigmaDetect"])
                #print(xImpsAllGroups[i,j,:])
                if np.isnan(row[j]): 
                  if np.isnan(pMissingGeomAvgGroups[i][priorGroup]): # IF THE GROUP MEAN IS NAN WE SET IT TO ZERO (parameter).
                      pMissingGeomAvgGroups[i][priorGroup] = 0
                  #print(pMissingGeomAvgGroups[i][priorGroup])
                  likelihood = pMissings * (1.0 - identPEP) * (1.0 - linkPEP) + pMissingGeomAvgGroups[i][priorGroup] * (identPEP * (1.0 - linkPEP) + linkPEP)
                else: #TRY TO UNDERSTAND THE RELATIONSHIP BETWEEN THIS AND HYPSEC DISTRIBUTION
                  if np.isnan(pMissingGeomAvgGroups[i][priorGroup]):
                      pMissingGeomAvgGroups[i][priorGroup] = 0 
                  likelihood = (1.0 - pMissings) * pDiffs[i,j,:] * (1.0 - identPEP) * (1.0 - linkPEP) + (1.0 - pMissingGeomAvgGroups[i][priorGroup]) * (pQuantIncorrectId[i][j] * identPEP * (1.0 - linkPEP) + linkPEP)
                if np.min(likelihood) == 0.0:
                  likelihood += np.nextafter(0,1)
        #likelihood = np.nan_to_num(likelihood)
                if np.isnan(likelihood).sum() > 0:
                    print("NaN count: " + str(np.isnan(likelihood).sum()))
                    print(likelihood)
                    raise ("NaN encountered in likelihood computations")
                pProteinQuant += np.log(likelihood)
        #pProteinQuant = np.nan_to_num(pProteinQuant) # fix NaN issue in protein quants
    pProteinQuant -= np.max(pProteinQuant)
    #print(pProteinQuant)
    pProteinQuant = np.exp(pProteinQuant) / np.sum(np.exp(pProteinQuant))
    pProteinQuantsList.append(pProteinQuant)
    
    #print(len(params["proteinQuantCandidates"]))
    eValue, confRegion = getPosteriorParams(params['proteinQuantCandidates'], pProteinQuant)
    #print(params['proteinQuantCandidates'])
    bayesQuantRow.append(eValue)
    
    ############################
    # FOR NON GROUPED ##########
    ############################
    
#    for i, row in enumerate(quantMatrix):
#      
#      linkPEP = quantRows[i].linkPEP[j]
#      identPEP = quantRows[i].identificationPEP[j]
#      if identPEP < 1.0:
#        pMissings = pMissing(xImpsAll[i,j,:], params["muDetect"], params["sigmaDetect"]) # Pr(f_grn = NaN | m_grn = 1, t_grn = 0)
#        
#        if np.isnan(row[j]):
#          likelihood = pMissings * (1.0 - identPEP) * (1.0 - linkPEP) + pMissingGeomAvg[i] * (identPEP * (1.0 - linkPEP) + linkPEP)
#        else: #TRY TO UNDERSTAND THE RELATIONSHIP BETWEEN THIS AND HYPSEC DISTRIBUTION
#          likelihood = (1.0 - pMissings) * pDiffs[i,j,:] * (1.0 - identPEP) * (1.0 - linkPEP) + (1.0 - pMissingGeomAvg[i]) * (pQuantIncorrectId[i][j] * identPEP * (1.0 - linkPEP) + linkPEP)
#        
#        if np.min(likelihood) == 0.0:
#          likelihood += np.nextafter(0,1)
#        #likelihood = np.nan_to_num(likelihood)
#        '''
#        print("""
#              DEBUG MESSAGE!!!!!!!!!!!!!!!!!
#              """)
#        print("pMissings")
#        print(pMissings)
#        print("pDiffs[i,j,:]")
#        print(pDiffs[i,j,:])
#        print("identPEP")
#        print(identPEP)
#        print("linkPEP")
#        print(linkPEP)
#        print("pMissingGeomAvg[i]")
#        print(pMissingGeomAvg[i])
#        print("pQuantIncorrectId[i][j]")
#        print(pQuantIncorrectId[i][j])
#        print("""
#              END OF MESSAGE!!!!!!!!!!
#              """)
#        '''
#        if np.isnan(likelihood).sum() > 0:
#            print("NaN count: " + str(np.isnan(likelihood).sum()))
#            print(likelihood)
#            raise ("NaN encountered in likelihood computations")
#        pProteinQuant += np.log(likelihood)
#        #pProteinQuant = np.nan_to_num(pProteinQuant) # fix NaN issue in protein quants
#      
#    pProteinQuant -= np.max(pProteinQuant)
#    #print(pProteinQuant)
#    pProteinQuant = np.exp(pProteinQuant) / np.sum(np.exp(pProteinQuant))
#    pProteinQuantsList.append(pProteinQuant)
#    
#    #print(len(params["proteinQuantCandidates"]))
#    eValue, confRegion = getPosteriorParams(params['proteinQuantCandidates'], pProteinQuant)
#    #print(params['proteinQuantCandidates'])
#    bayesQuantRow.append(eValue)
  
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
    pDiffPrior = params['inGroupDiffPrior'][groupId]
    if "shapeInGroupStdevs" in params:
      #pMu = getPosteriorProteinGroupMuMarginalized(filteredProteinQuantsList, params)
      pMu = getPosteriorProteinGroupMuMarginalized(pDiffPrior, filteredProteinQuantsList, params)
    else:
      pMu = getPosteriorProteinGroupMu(params['inGroupDiffPrior'], filteredProteinQuantsList, params)
    pProteinGroupQuants.append(pMu)
  
  return pProteinGroupQuants
  
def getPosteriorProteinGroupMu(pDiffPrior, pProteinQuantsList, params):
  pMus = np.zeros_like(params['proteinQuantCandidates'])
  for pProteinQuants in pProteinQuantsList:
    pMus += np.log(np.convolve(pDiffPrior, pProteinQuants, mode = 'valid'))
  
  #pMus = np.nan_to_num(pMus)
  pMus -= np.max(pMus)
  pMus = np.exp(pMus) / np.sum(np.exp(pMus))
  return pMus

def getPosteriorProteinGroupMuMarginalized(pDiffPrior, pProteinQuantsList, params):
  pMus = np.zeros((len(params['sigmaCandidates']), len(params['proteinQuantCandidates'])))
  for pProteinQuants in pProteinQuantsList:
    for idx, pDiffPrior in enumerate(params['inGroupDiffPrior']):
      pMus[idx,:] += np.log(np.convolve(pDiffPrior, pProteinQuants, mode = 'valid'))
  
  pSigmas = hyperparameters.funcGamma(params['sigmaCandidates'], params["shapeInGroupStdevs"], params["scaleInGroupStdevs"]) # prior
  pMus = np.log(np.dot(pSigmas, np.exp(pMus)))
  
  pMus -= np.max(pMus)
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

