from __future__ import print_function

import sys
import csv
import bisect
from collections import defaultdict

import numpy as np

from .. import parsers

def normalizeIntensitiesRtimeBased(clusterQuantExtraFile, clusterQuantExtraNormalizedFile, minRunsObservedIn, plotScatter = False, plotRunningAverage = False):
  featureGroups = parsers.parseFeatureClustersFile(clusterQuantExtraFile)
  factorPairs = getIntensityFactorPairs(featureGroups, sortKey = lambda x : (x.fileName, -1.0 * x.intensity, x.rTime), minRunsObservedIn = minRunsObservedIn)
  
  if plotScatter:
    plotFactorScatter(factorPairs)
  
  rTimeFactorArrays = getFactorArrays(factorPairs)
  
  if plotRunningAverage:
    plotFactorRunningAverage(rTimeFactorArrays)
  
  normalizeIntensitiesWithFactorArrays(clusterQuantExtraFile, rTimeFactorArrays, clusterQuantExtraNormalizedFile)

def getIntensityFactorPairs(featureGroups, sortKey, minRunsObservedIn):
  factorPairs = defaultdict(list)
  for i, featureGroup in enumerate(featureGroups):
    localFactorPairs = dict()
    sortedFeatures = sorted(featureGroup, key = sortKey)
    for row in sortedFeatures:
      fileName, intensity, rTime = sortKey(row)
      intensity *= -1.0
      if not np.isnan(intensity) and fileName not in localFactorPairs: # and row.charge == 2:
        localFactorPairs[fileName] = (np.log2(intensity), rTime)
    if len(localFactorPairs) >= minRunsObservedIn:
      keys = sorted(localFactorPairs.keys())
      masterKey = keys[0]
      factor0 = sum([localFactorPairs[masterKey][0] - x[0] for x in localFactorPairs.values()]) / len(keys)
      factorPairs[masterKey].append((localFactorPairs[masterKey][1], factor0))
      for key in keys[1:]:
        factori = factorPairs[masterKey][-1][1] - (localFactorPairs[masterKey][0] - localFactorPairs[key][0])
        factorPairs[key].append((localFactorPairs[key][1], factori))
    if (i+1) % 10000 == 0:
      print("Processing cluster", i+1)
  return factorPairs

# returns running averages of factors
def getFactorArrays(factorPairs, N = 2000):
  factorArrays = defaultdict(list)
  for i, key in enumerate(sorted(factorPairs.keys())):
    print("Calculating factor array", i+1)
    factorPairs[key] = sorted(factorPairs[key], key = lambda x : x[0])
    rTimes = [x[0] for x in factorPairs[key]]
    factors = [x[1] for x in factorPairs[key]]
    factorArrays[key] = zip(rTimes[int(N/2):int(-N/2)], runningMean(factors, N))
  return factorArrays

def runningMean(x, N):
  cumsum = np.cumsum(np.insert(x, 0, 0)) 
  return (cumsum[N:] - cumsum[:-N]) / N 
  
def normalizeIntensitiesWithFactorArrays(clusterQuantExtraFile, rTimeFactorArrays, clusterQuantExtraNormalizedFile):
  rTimeArrays, factorArrays = dict(), dict()
  for key in rTimeFactorArrays:
    rTimeArrays[key], factorArrays[key] = zip(*rTimeFactorArrays[key])
  print("Writing", clusterQuantExtraNormalizedFile)
  writer = parsers.getTsvWriter(clusterQuantExtraNormalizedFile)
  precClusters = parsers.parseFeatureClustersFile(clusterQuantExtraFile)
  for i, precCluster in enumerate(precClusters):
    for row in precCluster:
      outRow = list(row[1:])
      outRow[4] = getNormalizedIntensity(rTimeArrays[row.fileName], factorArrays[row.fileName], row.rTime, row.intensity)
      writer.writerow(outRow)
    writer.writerow([])
    if (i+1) % 10000 == 0:
      print("Writing cluster", i+1)

def getNormalizedIntensity(rTimeArray, factorArray, rTime, intensity):
  rTimeIndex = min([bisect.bisect_left(rTimeArray, rTime), len(rTimeArray) - 1])
  return intensity / (2 ** factorArray[rTimeIndex])

def plotFactorScatter(factorPairs):
  import scatter
  import matplotlib.pyplot as plt
  scatter.prepareSubplots()
  for i, key in enumerate(sorted(factorPairs.keys())):
    plt.subplot((len(factorPairs)-1) / 4 + 1,4,i+1)
    print("Calculating density", i+1)
    rTimes = [x[0] for x in factorPairs[key] if abs(x[1]) < 2]
    factors = [x[1] for x in factorPairs[key] if abs(x[1]) < 2]
    scatter.plotDensityScatterHist(rTimes, factors, bins = [20,20])
    plt.plot([min(rTimes),max(rTimes)],[0,0],'k:')
    plt.title(key.replace("JD_06232014_","").replace("_","\_"))
    plt.xlabel("Retention time")
    plt.ylabel("log2(int0/int1)")
    plt.ylim([-2,2])
  plt.show()

def plotFactorRunningAverage(factorArrays):
  import scatter
  import matplotlib.pyplot as plt
  scatter.prepareSubplots()
  for i, key in enumerate(sorted(factorArrays.keys())):
    plt.subplot((len(factorArrays)-1) / 4 + 1,4,i+1)
    plt.title(key.replace("JD_06232014_","").replace("_","\_"))
    plt.xlabel("Retention time")
    plt.ylabel("log2(int0/int1)")
    plt.ylim([-2,2])
    plt.plot(*zip(*factorArrays[key]))
  plt.show()

if __name__ == "__main__":
   main(sys.argv[1:])
