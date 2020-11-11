#!/usr/bin/python

from __future__ import print_function

import sys
import os
import warnings
import itertools
import textwrap
import collections

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, gamma, norm
from scipy.optimize import curve_fit

from ..triqler import __version__, __copyright__
from .. import parsers
from .. import hyperparameters
from .. import pgm
from .. import diff_exp

def main():
  print('Triqler.distribution.plot_posteriors version %s\n%s' % (__version__, __copyright__))
  print('Issued command:', os.path.basename(__file__) + " " + " ".join(map(str, sys.argv[1:])))
  
  args, params = parseArgs()
  
  params['warningFilter'] = "ignore"
  with warnings.catch_warnings():
    warnings.simplefilter(params['warningFilter'])
    plotPosterior(args.in_file, args.protein_id, args.protein_id_list, params)

def parseArgs():
  import argparse
  apars = argparse.ArgumentParser(
      formatter_class=argparse.RawDescriptionHelpFormatter,
      description=textwrap.dedent('''Plots posterior distribution for a given protein <P>. 
      
There are two different options for <IN_FILE>: 
1. use one of the posterior output files of Triqler, generated with the 
   --write_<mode>_posteriors flags. This will only print the particular 
   mode (protein, group or fold change).
2. use a Triqler input file. This estimates hyperparameters from the 
   entire input file and subsequently computes and plots posteriors from 
   all modes.
'''))

  apars.add_argument('in_file', default=None, metavar = "IN_FILE",
                     help='Triqler distribution output file or Triqler input file')
  
  apars.add_argument('--protein_id', metavar='P', 
                     help='Protein ID, can be a partial match, e.g. "P39744" will match "sp|P39744|NOC2_YEAST"')
  
  apars.add_argument('--protein_id_list', metavar='L', 
                     help='Text file with a list of protein IDs (one per line). The IDs have to be written exactly as specified in the input file (no partial matches). Specifying this flag will result in a posterior heatmap plot.')
  
  apars.add_argument('--fold_change_eval', type=float, default=1.0, metavar='F',
                     help='log2 fold change evaluation threshold')
                     
  apars.add_argument('--decoy_pattern', default = "decoy_", metavar='D', 
                     help='Prefix for decoy proteins (only when Triqler input file is used as input)')
  
  apars.add_argument('--plot_max_fold_change', type=float, default = 2.0, metavar='M', 
                     help='Maximum (absolute) fold change for the violin and posterior heatmap plots.')
  
  apars.add_argument('--plot_max_prob', type=float, default = 0.2, metavar='P', 
                     help='Maximum probability for the violin plots')
  
  apars.add_argument('--hide_labels',
                     help='Hide protein labels in the posterior heatmap plot.',
                     action='store_true')
  
  # ------------------------------------------------
  args = apars.parse_args()
  
  if args.protein_id and args.protein_id_list:
    sys.exit("ERROR: please specify either --protein_id or --protein_id_list, not both")
  elif not args.protein_id and not args.protein_id_list:
    sys.exit("ERROR: please specify either --protein_id or --protein_id_list")
    
  params = dict()
  params['returnPosteriors'] = True
  params["foldChangeEval"] = args.fold_change_eval
  params["decoyPattern"] = args.decoy_pattern
  params["trueConcentrationsDict"] = dict()
  params['pMax'] = args.plot_max_prob # max probability in violin plots
  params['maxFoldChange'] = args.plot_max_fold_change # max fold change in violin plots
  params['hideProteinLabels'] = args.hide_labels
    
  return args, params
  
def plotPosterior(inputFile, protein, proteinList, params):
  with open(inputFile, 'r') as f:
    line = f.readline()
    headerCols = line.split('\t')
    if len(headerCols) < 2:
      sys.exit("Could not identify input format.")
    elif headerCols[0] == 'run':
      plotPosteriorFromTriqlerInput(inputFile, protein, proteinList, params)
    elif headerCols[1] == 'group:run':
      if proteinList:
        sys.exit("Protein list posterior plotting not yet supported for protein posteriors")
      params['proteinQuantCandidates'] = np.array(list(map(float, headerCols[2:])))
      plotProteinPosteriors(inputFile, protein, proteinList, params)
    elif headerCols[1] == 'group':
      if proteinList:
        sys.exit("Protein list posterior plotting not yet supported for group posteriors")
      params['proteinQuantCandidates'] = np.array(list(map(float, headerCols[2:])))
      plotGroupPosteriors(inputFile, protein, proteinList, params)
    elif headerCols[1] == 'comparison':
      if proteinList:
        sys.exit("Protein list posterior plotting not yet supported for fold change posteriors")
      params['proteinDiffCandidates'] = np.array(list(map(float, headerCols[2:])))
      plotFoldChangePosteriors(inputFile, protein, proteinList, params)
    else:
      sys.exit("Could not identify input format.")
  
  plt.show()

def plotPosteriorFromTriqlerInput(triqlerInputFile, protein, proteinListFile, params):
  if not os.path.isfile(triqlerInputFile):
    sys.exit("Could not locate input file %s. Check if the path to the input file is correct." % triqlerInputFile)
  
  peptQuantRowFile = triqlerInputFile + ".pqr.tsv"
  if not os.path.isfile(peptQuantRowFile):
    sys.exit("Could not locate peptide quantification file %s. Run triqler to generate this file." % peptQuantRowFile)
  
  params["runIds"], params['groups'], params['groupLabels'], peptQuantRows = parsers.parsePeptideQuantFile(peptQuantRowFile)

  matplotlib.rcParams['axes.unicode_minus'] = False
  
  print("Fitting hyperparameters")
  hyperparameters.fitPriors(peptQuantRows, params)
  print("")
  
  peptidePEPThreshold = getPeptidePEPThreshold(peptQuantRows) # needed for the naive method
  
  if protein:
    peptQuantRows = list(filter(lambda x : protein in x.protein[0], peptQuantRows))
    if len(peptQuantRows) == 0:
      sys.exit("Could not find any peptides for the protein")
    
    protQuantRows = parsers.filterAndGroupPeptides(peptQuantRows, lambda x : not x.protein[0].startswith(params['decoyPattern']))
    
    plottedProtein = plotPosteriors(protQuantRows, params)
    
    # plots second half of fold change violin plots with the naive quant method
    #peptQuantRows = list(filter(lambda x : plottedProtein == x.protein[0], peptQuantRows))
    #plotPosteriorCalibration(peptQuantRows, peptidePEPThreshold, params, protein)
    
    finishViolinPlots(params, protein, rotate = False)
  else:
    numGroups = len(params['groups'])
    with open(proteinListFile, 'r') as f:
      proteinList = f.read().splitlines()

    peptQuantRows = list(filter(lambda x : x.protein[0] in proteinList, peptQuantRows))
    if len(peptQuantRows) == 0:
      sys.exit("Could not find any peptides for proteins matching to the input list")
    
    protQuantRows = parsers.filterAndGroupPeptides(peptQuantRows, lambda x : not x.protein[0].startswith(params['decoyPattern']))
    
    #protQuantRows = sorted(protQuantRows, key = lambda prot, quantRows: np.prod([x.combinedPEP for x in quantRows]), reverse = True)
    bayesDists = collections.defaultdict(dict)
    for prot, quantRows in protQuantRows:      
      quantRows, quantMatrix = parsers.getQuantMatrix(quantRows)
      
      proteinQuantIdPEP = np.prod([x.combinedPEP for x in quantRows])
      
      bayesQuantRow, _, _, posteriorDists = pgm.getPosteriors(quantRows, params)
      pProteinQuantsList, pProteinGroupQuants, pProteinGroupDiffs = posteriorDists
      
      for groupId1, groupId2 in itertools.combinations(range(numGroups), 2):
        x = params['proteinDiffCandidates']
        leftIdx, rightIdx = np.searchsorted(np.log2(np.power(10, x)), -1*params['maxFoldChange'], side = 'left'), np.searchsorted(np.log2(np.power(10, x)), params['maxFoldChange'], side = 'right')
        x = x[leftIdx:rightIdx]
        
        key = proteinQuantIdPEP
        bayesDists[(groupId1, groupId2)][(key, prot)] = pProteinGroupDiffs[(groupId1, groupId2)][leftIdx:rightIdx]
    
    trueLogRatio = None
    for plotIdx, (groupId1, groupId2) in enumerate(itertools.combinations(range(numGroups), 2)):
      plt.figure(figsize = (8,10))
    
      sortedKeys, bayesMatrix = sortMatrix(bayesDists[(groupId1, groupId2)])
      
      plotPosteriorsHeatMap(bayesMatrix, sortedKeys, trueLogRatio, params)
      
      if 'groupLabels' in params:
        title = "f.c. posteriors %s vs %s" % (params['groupLabels'][groupId1], params['groupLabels'][groupId2])
      else:
        title = "f.c. posteriors %dvs%d" % (groupId1 + 1, groupId2 + 1)
      plt.title(title, fontsize = 24)
      plt.tight_layout()

def sortMatrix(distDict):
  sortedKeys = sorted(distDict.keys())
  return sortedKeys, np.matrix([distDict[key] for key in sortedKeys])
      
def plotPosteriorsHeatMap(distributionMatrix, sortedKeys, trueLogRatio, params, cmap = matplotlib.cm.Blues):
  cs = np.cumsum(distributionMatrix, axis = 1)
  csRev = np.cumsum(distributionMatrix[:,::-1], axis = 1)
  s = np.sum(distributionMatrix, axis = 1)
  distributionMatrix[(cs >= 0.45 * s) & ((csRev >= 0.45 * s)[:,::-1])] = 10*params['pMax']
  distributionMatrix[(cs < 0.45 * s) | ((csRev < 0.45 * s)[:,::-1])] = 50*params['pMax']
  distributionMatrix[(cs < 0.025 * s) | ((csRev < 0.025 * s)[:,::-1])] = 0.0
  #distributionMatrix[(distributionMatrix > 0) & (distributionMatrix < 0.95*np.max(distributionMatrix, axis = 1))] = 50*params['pMax']
  
  cmap.set_bad('black',0.6)
  cmap.set_under('black',1.0)
  
  norm = matplotlib.colors.Normalize(clip = False)
  
  plt.plot([0,0], [0, len(distributionMatrix)], 'w:', alpha = 0.6)
  if trueLogRatio:
    if np.abs(trueLogRatio) < params['maxFoldChange']:
      plt.plot([trueLogRatio,trueLogRatio], [0, len(distributionMatrix)], 'w--', label = 'true ratio')
    else:
      numMarkers = 50
      if trueLogRatio < 0:
        sign, marker = -1, matplotlib.markers.CARETLEFT
      else:
        sign, marker = 1, matplotlib.markers.CARETRIGHT
      plt.plot([sign*0.95*params['maxFoldChange']]*numMarkers, np.linspace(0, len(distributionMatrix), numMarkers), 'w', linestyle = 'None', marker = marker, label = 'true ratio')
  
  plt.fill_betweenx([0, len(distributionMatrix)], [-1*params['foldChangeEval'], -1*params['foldChangeEval']], [params['foldChangeEval'], params['foldChangeEval']], color = 'red', alpha = 0.3, label = 'fold change threshold')
  
  plt.imshow(distributionMatrix, aspect = 'auto', interpolation = 'none', norm = norm, vmin = 1e-3, vmax = 100*params['pMax'], extent = (-1*params['maxFoldChange'], params['maxFoldChange'], 0, len(distributionMatrix)), cmap = cmap)
  
  if params['hideProteinLabels']:
    plt.gca().set_yticklabels([])
    plt.gca().yaxis.set_ticks_position('none')
  else:
    plt.gca().set_yticklabels([x[1] for x in sortedKeys][::-1])
    plt.gca().set_yticks(np.arange(len(sortedKeys))+0.5)
  
  for tick in plt.gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(18)
  plt.xlabel("log2(fold change)", fontsize = 20)
    
def getPeptidePEPThreshold(peptQuantRows):
  sumPEP = 0.0
  peptidePEPThreshold = 1.0
  for i, pqr in enumerate(filter(lambda x : not x.protein[0].startswith("decoy_"), sorted(peptQuantRows, key = lambda x : x.combinedPEP))):
    sumPEP += pqr.combinedPEP
    peptidePEPThreshold = pqr.combinedPEP
    if sumPEP / (i+1.0) > 0.05:
      break
  return peptidePEPThreshold

def getNaivePosteriorParams(peptQuantRows, peptidePEPThreshold, params, minQuant = 0.05): 
  quantRows = sorted(filter(lambda x : x.combinedPEP <= peptidePEPThreshold, peptQuantRows), key = lambda x : -1 * np.sum(x.quant))
  seenPeptides = set()
  usablePeptides = 0
  filteredQuantRows = list()
  for quantRow in quantRows:
    if quantRow.peptide not in seenPeptides:
      seenPeptides.add(quantRow.peptide)
      filteredQuantRows.append(quantRow.quant)
  
  #print(len(seenPeptides), "unique peptides")
  
  if len(filteredQuantRows) < 3: # use top3 for protein summarization
    print("Skipping naive quant because < 3 peptides were identified")
    return dict(), dict(), seenPeptides
  
  quant = np.zeros(len(filteredQuantRows[0]))
  for quantRow in filteredQuantRows[:3]: # use top3 for protein summarization
    q = np.array(quantRow)
    q[q == 0] = np.mean(q[q != 0]) # missing value imputation by row mean
    #q[q == 0] = minQuant # missing value imputation by limit of quantification
    quant += q
  quant = parsers.geoNormalize(quant)
  quant = np.log2(quant)
  
  numGroups = len(params["groups"])
  
  naiveRatioMu, naiveRatioSigma = dict(), dict()
  for groupId1, groupId2 in itertools.combinations(range(numGroups), 2):
    naiveRatioMu[(groupId1, groupId2)] = np.mean([quant[x] for x in params['groups'][groupId1]]) - np.mean([quant[x] for x in params['groups'][groupId2]])
    naiveRatioSigma[(groupId1, groupId2)] = np.sqrt(np.var([quant[x] for x in params['groups'][groupId1]]) / len(params['groups'][groupId1]) + np.var([quant[x] for x in params['groups'][groupId2]])  / len(params['groups'][groupId2]) )
  
  return naiveRatioMu, naiveRatioSigma, seenPeptides
    
def plotPosteriorCalibration(peptQuantRows, peptidePEPThreshold, params, protein):
  naiveRatioMu, naiveRatioSigma, seenPeptides = getNaivePosteriorParams(peptQuantRows, peptidePEPThreshold, params)
  
  numGroups = len(params["groups"])
  
  plotIdx = 1
  plt.figure(3)
  for groupId1, groupId2 in itertools.combinations(range(numGroups), 2):
    plt.subplot(1, ((numGroups - 1)*numGroups)/2, plotIdx)
    
    x = np.arange(-1*params['maxFoldChange'],params['maxFoldChange'],0.01)
    
    if len(naiveRatioMu) > 0: # use top3 for protein summarization
      naiveDist = -1.0*norm.pdf(x, naiveRatioMu[(groupId1, groupId2)], naiveRatioSigma[(groupId1, groupId2)]) / 100 * np.log2(10)  
      plotViolin(naiveDist, x, label = 'Naive quant', color = 'blue')
    #plt.fill_between(x, naiveDist*0.0, naiveDist, where = np.abs(x) > params['foldChangeEval'], facecolor = 'red', alpha = 0.5)
    
    plotIdx += 1

def finishViolinPlots(params, protein, rotate = False):
  trueConcentrations = diff_exp.getTrueConcentrations(params["trueConcentrationsDict"], protein)
  
  numGroups = len(params["groups"])
  if len(trueConcentrations) == 0:
    trueConcentrations = np.array([1.0]*numGroups)
  
  trueRatios = np.dot(np.matrix(trueConcentrations).T, np.ones_like(np.matrix(trueConcentrations)))
  trueLogRatios = np.log2(trueRatios / trueRatios.T)
  
  plotIdx = 1
  plt.figure(3)
  for groupId1, groupId2 in itertools.combinations(range(numGroups), 2):
    realRatio = trueLogRatios[groupId1,groupId2]
    
    plt.subplot(1, ((numGroups - 1)*numGroups)/2, plotIdx)
    
    #ymin, ymax = -1*params['pMax'], params['pMax']
    ymin, ymax = 0, params['pMax']
    
    x = np.arange(-1*params['maxFoldChange'],params['maxFoldChange'],0.01)
    
    if len(params["trueConcentrationsDict"]) > 0:
      if np.abs(realRatio) < params['maxFoldChange']:
        if rotate:
          plt.plot([ymin, ymax], [realRatio, realRatio], 'k--', label = 'true ratio')
        else:
          plt.plot([realRatio, realRatio], [ymin, ymax], 'k--', label = 'true ratio')
      else:
        numMarkers = 10
        if realRatio < 0:
          sign, marker = -1, matplotlib.markers.CARETDOWN
        else:
          sign, marker = 1, matplotlib.markers.CARETUP
        if rotate:
          plt.plot(np.linspace(ymin, ymax, numMarkers), [sign*0.95*params['maxFoldChange']]*numMarkers, 'k--', marker = marker, label = 'true ratio')
        else:
          plt.plot([sign*0.95*params['maxFoldChange']]*numMarkers, np.linspace(ymin, ymax, numMarkers), 'k--', marker = marker, label = 'true ratio')
    
    if rotate:
      plt.plot([ymin, ymax], [-1*params['foldChangeEval'], -1*params['foldChangeEval']], color = 'red', alpha = 0.5, label = 'fold change threshold')
      plt.plot([ymin, ymax], [params['foldChangeEval'], params['foldChangeEval']], color = 'red', alpha = 0.5)
      plt.fill_between([ymin, ymax], [-1*params['foldChangeEval'], -1*params['foldChangeEval']], [params['foldChangeEval'], params['foldChangeEval']], color = 'red', alpha = 0.15)
      plt.xlim([ymin, ymax])
      plt.ylim([-1*params['maxFoldChange'], params['maxFoldChange']])
    else:
      plt.plot([-1*params['foldChangeEval'], -1*params['foldChangeEval']], [ymin, ymax], color = 'red', alpha = 0.5, label = 'fold change threshold')
      plt.plot([params['foldChangeEval'], params['foldChangeEval']], [ymin, ymax], color = 'red', alpha = 0.5)
      plt.fill_betweenx([ymin, ymax], [-1*params['foldChangeEval'], -1*params['foldChangeEval']], [params['foldChangeEval'], params['foldChangeEval']], color = 'red', alpha = 0.15)
      plt.ylim([ymin, ymax])
      plt.xlim([-1*params['maxFoldChange'], params['maxFoldChange']])
    
    
    if plotIdx == 1:
      if rotate:
        plt.ylabel("log2(fold change)", fontsize = 14)
        plt.gca().yaxis.set_ticks_position('left')
        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        #plt.gca().spines['right'].set_visible(False)
      else:
        plt.xlabel("log2(fold change)", fontsize = 14)
        plt.gca().xaxis.set_ticks_position('bottom')
        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        #plt.gca().spines['right'].set_visible(False)
    else:
      if rotate:
        plt.gca().set_yticklabels([])
        plt.gca().yaxis.set_ticks_position('none')
      else:
        plt.gca().set_xticklabels([])
        plt.gca().xaxis.set_ticks_position('none')
    
    if rotate:
      plt.gca().xaxis.set_ticks_position('none') 
      plt.gca().set_xticklabels([])
    else:
      plt.gca().yaxis.set_ticks_position('none') 
      plt.gca().set_yticklabels([])
      
    plotIdx += 1
  
  plt.tight_layout()
  #plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # matplotlib < 2.2
  #plt.subplots_adjust(wspace=0, hspace=0)
  plt.legend(loc = 'upper right', fontsize = 14)
  
def plotPosteriors(protQuantRows, params):
  numGroups = len(params['groups'])
  
  imputedDiffs, observedXICValues = list(), list()
  plottedProtein = ""
  for idx, (prot, quantRows) in enumerate(protQuantRows):
    if idx > 0:
      print("WARNING: Found more than 1 protein matching the --protein_id flag. Only plotting results for the first hit.\n         Use the --protein_id_list flag to plot fold change posteriors for multiple proteins at once.")
      break
    else:
      plottedProtein = prot
    
    quantRows, quantMatrix = parsers.getQuantMatrix(quantRows)
    
    print("Protein ID:", prot)
    print("")
    
    print("Peptide absolute abundances")
    printQuantRows(quantMatrix, quantRows)

    quantMatrixNormalized = [parsers.geoNormalize(row) for row in quantMatrix]
    print("Peptide relative abundances")
    printQuantRows(quantMatrixNormalized, quantRows)
    
    #geoAvgQuantRow = getSemiNaiveQuants(quantMatrixNormalized, quantRows)
    
    #print("Protein abundance and p-value semi naive quant")
    #printStats(geoAvgQuantRow, params['groups'])
    
    bayesQuantRow, _, probsBelowFoldChange, posteriorDists = pgm.getPosteriors(quantRows, params)
    pProteinQuantsList, pProteinGroupQuants, pProteinGroupDiffs = posteriorDists
    
    print("Protein abundance (expected value) and p-value")
    printStats(bayesQuantRow, params['groups'])
    
    print("Posterior probability |log2 fold change| < %.2f" % (params['foldChangeEval']))
    if numGroups >= 2:
      for groupId1, groupId2 in itertools.combinations(range(numGroups), 2):
        print("  Group %s vs Group %s: %f" % (params['groupLabels'][groupId1], params['groupLabels'][groupId2], probsBelowFoldChange[(groupId1, groupId2)]))
    #print("Probability |log2 fold change| < %.2f:" % (params['foldChangeEval']), probsBelowFoldChange['ANOVA'])
    print("")
    
    plt.figure(1)
    plotPosteriorProteinRatios(pProteinQuantsList, params)
    
    plt.figure(2)
    plotPosteriorProteinGroupsRatios(pProteinGroupQuants, params)
    
    plt.figure(3)
    plotPosteriorProteinGroupsDiffs(pProteinGroupDiffs, params)
    
    #plt.figure(5)
    #q = geoAvgQuantRow
    #q = bayesQuantRow
    #data = [[np.log10(q[x]) for x in group if not np.isnan(q[x])] for group in params['groups']]
    #plt.boxplot(data)
  return plottedProtein

def printQuantRows(quantMatrix, quantRows):
  for i, row in enumerate(quantMatrix):
    print("\t".join(['%.2f' % x for x in quantMatrix[i]]) + '\tcombinedPEP=%.2g\tpeptide=%s' % (quantRows[i].combinedPEP, quantRows[i].peptide))
  print("")
  
def printStats(geoAvgQuantRow, groups):
  print("\t".join(['%.2f' % x for x in geoAvgQuantRow]))
  
  args = parsers.getQuantGroups(geoAvgQuantRow, groups)
  anovaFvalue, anovaPvalue = f_oneway(*args)
  print("p-value:", anovaPvalue)
  print("")
  
  float_formatter = lambda x: "%.2f" % x
  np.set_printoptions(formatter={'float_kind':float_formatter})
  
  geoAvgs = np.matrix([parsers.geomAvg([2**y for y in x]) for x in args])
  ratioMatrix = np.log2(np.transpose(geoAvgs) / geoAvgs)
  #print(ratioMatrix)
  #print("")

def getSemiNaiveQuants(quantMatrixNormalized, quantRows):
  numSamples = len(quantMatrixNormalized[0])
  geoAvgQuantRow = [0]*numSamples
  for i in range(numSamples):
    geoAvgQuantRow[i] = parsers.weightedGeomAvg([x[i] for x in quantMatrixNormalized], [1.0 - y.combinedPEP if not np.isnan(x[i]) else np.nan for x, y in zip(quantMatrixNormalized, quantRows)])
  #geoAvgQuantRow = geoNormalize(geoAvgQuantRow)
  geoAvgQuantRow = np.array(geoAvgQuantRow)
  return geoAvgQuantRow
  
def plotPosteriorProteinGroupsDiffs(pProteinGroupDiffs, params):
  numGroups = len(params['groups'])  
  plt.suptitle("Posteriors for fold change differences between groups", fontsize = 14)
  
  plotIdx = 1
  for groupId1, groupId2 in itertools.combinations(range(numGroups), 2):
    pDifference = pProteinGroupDiffs[(groupId1, groupId2)]
    
    #plt.subplot(numGroups - 1, numGroups - 1, groupId1 * (numGroups - 1) + (groupId2 - 1) + 1)
    plt.subplot(1, ((numGroups - 1)*numGroups)/2, plotIdx)
    plotIdx += 1
    if 'groupLabels' in params:
      title = "%s vs %s" % (params['groupLabels'][groupId1], params['groupLabels'][groupId2])
    else:
      title = "%dvs%d" % (groupId1 + 1, groupId2 + 1)
    plt.title(title, fontsize = 14)
    log2diff = np.log2(np.power(10, params['proteinDiffCandidates']))
    plotViolin(pDifference, log2diff, 'Triqler', 'green')
    #plt.fill_between(log2diff, pDifference*0.0, pDifference, where = np.abs(log2diff) > params['foldChangeEval'], facecolor = 'red', alpha = 0.5)
    
    #if groupId1 + 1 == groupId2: # only print(x label on diagonal plots)
    #  plt.ylabel("log2(protein quant group%d / protein quant group%d)" % (groupId1 + 1, groupId2 + 1))
    plt.ylim([-1*params['maxFoldChange'], params['maxFoldChange']])
    #plt.ylim([np.floor(log2diff[np.argmax(pDifference > 1e-4)]), np.ceil(log2diff[::-1][np.argmax(pDifference[::-1] > 1e-4)])])

def plotViolin(pDifference, log2diff, label, color, rotate = False):
  log2diff = log2diff[np.where(np.abs(pDifference) > 1e-4)]
  pDifference = pDifference[np.where(np.abs(pDifference) > 1e-4)]
  
  if rotate:
    plt.plot(pDifference, log2diff, color = color)
    plt.fill_betweenx(log2diff, pDifference, alpha = 0.5, label = label, color = color)
  else:
    plt.plot(log2diff, pDifference, color = color)
    plt.fill_between(log2diff, pDifference*0.0, pDifference, alpha = 0.5, label = label, color = color)
    
def plotPosteriorProteinGroupsRatios(pProteinGroupQuants, params):
  minProteinRatio, maxProteinRatio = max(params['proteinQuantCandidates']), min(params['proteinQuantCandidates'])
  
  print("Normal distribution fits for posterior distributions of treatment group relative abundances:")
  plt.suptitle("Posteriors for treatment group abundances", fontsize = 14)
  numGroups = len(params['groups'])
  for groupId in range(numGroups):
    plt.subplot(numGroups, 1, groupId + 1)
    plt.title("Group %s" % (params['groupLabels'][groupId]))
    plt.plot(params['proteinQuantCandidates'], pProteinGroupQuants[groupId], label='posterior')
    
    # fit protein group ratios with normal distribution
    popt, pcov = curve_fit(lambda x, mu, sigma : norm.pdf(x, mu, sigma), params['proteinQuantCandidates'], pProteinGroupQuants[groupId] * 100)
    mu, sigma = popt
    
    varNames = ["mu", "sigma"]
    outputString = "  Group %s: " % (params['groupLabels'][groupId]) + ", ".join(["%s"]*len(popt)) + " = " + ", ".join(["%f"] * len(popt))
    print(outputString % tuple(varNames + list(popt)))
    
    plt.plot(params['proteinQuantCandidates'], norm.pdf(params['proteinQuantCandidates'], mu, sigma) / 100, ':', label='normal fit')
    
    minProteinRatio = min([minProteinRatio, params['proteinQuantCandidates'][np.argmax(pProteinGroupQuants[groupId] > 1e-4)]])
    maxProteinRatio = max([maxProteinRatio, params['proteinQuantCandidates'][::-1][np.argmax(pProteinGroupQuants[groupId][::-1] > 1e-4)]])
    
  for groupId in range(numGroups):
    plt.subplot(numGroups, 1, groupId + 1)
    plt.xlim([np.floor(minProteinRatio), np.ceil(maxProteinRatio)])
    if groupId == 0:
      plt.legend()
    if groupId == numGroups - 1:
      plt.xlabel('log10(rel. protein quant)', fontsize = 14)
  plt.tight_layout()
  #plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # matplotlib < 2.2

def plotPosteriorProteinRatios(pProteinQuantsList, params, maxCols = 15):
  minProteinRatio, maxProteinRatio = max(params['proteinQuantCandidates']), min(params['proteinQuantCandidates'])
  cols = min([len(g) for g in params['groups']] + [maxCols])
  rows = len(params['groups'])
  
  plt.suptitle("Posteriors for protein abundances", fontsize = 14)
  for j, pProteinQuants in enumerate(pProteinQuantsList):
    if len([g for g, x in enumerate(params['groups']) if j in x]) > 0:
      groupId = [g for g, x in enumerate(params['groups']) if j in x][0]
    else:
      continue
    
    inGroupIdx = params['groups'][groupId].index(j)
    if inGroupIdx < cols:
      plt.subplot(rows, 1, groupId + 1)
      if inGroupIdx < 5:
        label = params['runIds'][j]
      else:
        label = None
      plt.plot(params['proteinQuantCandidates'], pProteinQuants, label = label)
      minProteinRatio = min([minProteinRatio, params['proteinQuantCandidates'][np.argmax(pProteinQuants > 1e-4)]])
      maxProteinRatio = max([maxProteinRatio, params['proteinQuantCandidates'][::-1][np.argmax(pProteinQuants[::-1] > 1e-4)]])
    
  for groupId in range(rows):
    plt.subplot(rows, 1, groupId + 1)
    plt.title("Group %s" % (params['groupLabels'][groupId]))
    plt.xlim([np.floor(minProteinRatio), np.ceil(maxProteinRatio)])
    plt.legend()
    if groupId == rows - 1:
      plt.xlabel('log10(rel. protein quant)', fontsize = 14)
  plt.tight_layout()
  #plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # matplotlib < 2.2

def plotProteinPosteriors(inputFile, protein, params):
  pProteinQuantsList = list()
  params['groupLabels'] = list()
  params['groups'] = list()
  params['runIds'] = list()
  for fileIdx, (protein, groupRun, posterior) in enumerate(parsers.parsePosteriorFile(inputFile, refProtein = protein)):
    pProteinQuantsList.append(posterior)
    group = ":".join(groupRun.split(":")[:-1])
    run = groupRun.split(":")[-1]
    groupIdx = addGroup(group, params)
    params['groups'][groupIdx].append(fileIdx)
    params['runIds'].append(run)
  
  plotPosteriorProteinRatios(pProteinQuantsList, params)
  
def plotGroupPosteriors(inputFile, protein, params):
  pProteinGroupQuants = list()
  params['groupLabels'] = list()
  params['groups'] = list()
  for protein, group, posterior in parsers.parsePosteriorFile(inputFile, refProtein = protein):
    pProteinGroupQuants.append(posterior)
    addGroup(group, params)
  
  plotPosteriorProteinGroupsRatios(pProteinGroupQuants, params)
  
def plotFoldChangePosteriors(inputFile, protein, params):
  pProteinGroupDiffs = dict()
  params['groupLabels'] = list()
  params['groups'] = list()
  params['maxFoldChange'] = 0.0
  for protein, comparison, posterior in parsers.parsePosteriorFile(inputFile, refProtein = protein):
    group1, group2 = comparison.split("_vs_")
    groupId1 = addGroup(group1, params)
    groupId2 = addGroup(group2, params)
    pProteinGroupDiffs[(groupId1, groupId2)] = posterior
  
    log2diff = np.log2(np.power(10, params['proteinDiffCandidates']))
    params['maxFoldChange'] = max([params['maxFoldChange'], np.abs(np.floor(log2diff[np.argmax(posterior > 1e-4)]))])
    params['maxFoldChange'] = max([params['maxFoldChange'], np.abs(np.ceil(log2diff[::-1][np.argmax(posterior[::-1] > 1e-4)]))])
  
  plotPosteriorProteinGroupsDiffs(pProteinGroupDiffs, params)

def addGroup(group, params):
  if group not in params['groupLabels']:
      params['groupLabels'].append(group)
      params['groups'].append([])
  return params['groupLabels'].index(group)

if __name__ == "__main__":
   main()
