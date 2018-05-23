#!/usr/bin/python

'''
Some helper functions to parse custom data files
'''

from __future__ import print_function

import sys
import numpy as np
import math
import csv
import os
from collections import defaultdict
from collections import namedtuple
import itertools

from . import percolator

####################################
## input: filename (one per line) ##
####################################

def parseFileList(inputFile):
  reader = csv.reader(open(inputFile, 'r'), delimiter = '\t')
  fileList = list()
  groups = list()
  groupNames = list()
  for fileIdx, row in enumerate(reader):
    fileList.append(row[0])
    if row[1] not in groupNames:
      groupNames.append(row[1])
      groups.append([])
    groups[groupNames.index(row[1])].append(fileIdx)
  
  for groupIdx, groupName in enumerate(groupNames):
    groupNames[groupIdx] = str(groupIdx + 1) + ":" + groupName
  return fileList, groups, groupNames

def getGroupLabel(idx, groups, groupLabels):
  groupIdx = [i for i, g in enumerate(groups) if idx in g][0]
  return groupLabels[groupIdx]

def getQuantGroups(quantRow, groups, transform = np.log2):
  args = list()
  for group in groups:
    args.append([transform(quantRow[i]) for i in group if not np.isnan(quantRow[i])])
  return args
  
#####################
## input: MS2 file ##
#####################

def parseMs2ToIntensityRTime(inputFile):
  scannrToIntensityRTimeMap = defaultdict(dict)

  scannr = 0
  with open(inputFile, 'r') as f:
    for line in f:
      if line.startswith('S'):
        scannr = int(line.split('\t')[1])
      elif line.startswith('I\tEZ'):
        s = line.split('\t')
        scannrToIntensityRTimeMap[scannr][(int(s[2]), float(s[3]))] = (float(s[5]), float(s[4]))

  return scannrToIntensityRTimeMap

PrecursorCandidate = namedtuple("PrecursorCandidate", "fileIdx, fileName precMz charge rTime intensity peptLinkPEPs")

def parseFeatureClustersFileHandle(reader):
  rows = list()
  name = ""
  for row in reader:
    if len(row) > 5:
      if float(row[4]) > 0.0:
        rows.append(PrecursorCandidate(-1, row[0], float(row[1]), int(row[2]), float(row[3]), float(row[4]), row[5]))
    elif len(row) == 1:
      name = row[0]
    else:
      if len(rows) > 0:
        yield rows
      rows = list()

def parseFeatureClustersFile(clusterQuantFile):
  reader = csv.reader(open(clusterQuantFile, 'r'), delimiter = '\t')
  i = 0
  for x in parseFeatureClustersFileHandle(reader):
    #if i > 10000:
    #  break
    i += 1
    yield x

TriqlerInputRowHeaders = "run condition charge spectrumId linkPEP featureClusterId searchScore intensity peptide proteins".split(" ")
TriqlerInputRowBase = namedtuple("TriqlerInputRow", TriqlerInputRowHeaders)

class TriqlerInputRow(TriqlerInputRowBase):
  def toList(self):
    l = list(self)
    return l[:-1] + l[-1]

  def toString(self):
    return "\t".join(map(str, self.toList()))

def parseTriqlerInputFile(triqlerInputFile):
  reader = csv.reader(open(triqlerInputFile, 'r'), delimiter = '\t')
  next(reader)
  for row in reader:
    yield TriqlerInputRow(row[0], row[1], int(row[2]), int(row[3]), float(row[4]), int(row[5]), float(row[6]), float(row[7]), row[8], row[9:])
  
PeptideQuantRowHeaders = "combinedPEP linkPEP expMass charge rTime pseudoPeptide quant protein identificationPEP peptide".split(" ")
PeptideQuantRowBase = namedtuple("PeptideQuantRow", PeptideQuantRowHeaders)

class PeptideQuantRow(PeptideQuantRowBase):
  def toList(self):
    l = list(self)
    return l[:1] + list(map(lambda x : '%.5g' % x, l[1])) + l[2:6] + list(map(lambda x : '%.2f' % x, l[6])) + l[7:]

  def toString(self):
    return "\t".join(map(str, self.toList()))

def getPeptideQuantRowHeaders(runs):
  return PeptideQuantRowHeaders[:1] + runs + PeptideQuantRowHeaders[2:6] + runs + PeptideQuantRowHeaders[7:]

def parsePeptideQuantFile(peptideQuantFile):
  reader = csv.reader(open(peptideQuantFile, 'r'), delimiter = '\t')
  header = next(reader)
  numRuns = header.index("protein") - header.index("pseudoPeptide") - 1
  peptideQuantRows = list()
  for row in reader:
    peptideQuantRows.append(PeptideQuantRow(float(row[0]), list(map(float, row[1:1+numRuns])), float(row[1+numRuns]), int(row[2+numRuns]), float(row[3+numRuns]), row[4+numRuns], list(map(float, row[5+numRuns:5+2*numRuns])), row[5+2*numRuns], float(row[6+2*numRuns]), row[7+2*numRuns]))

  runIdsWithGroup = header[1:1+numRuns]
  maxGroups = max([int(runId.split(":")[0]) for runId in runIdsWithGroup])
  runIds = list()
  groups, groupLabels = [None] * maxGroups, [None] * maxGroups
  for fileIdx, runId in enumerate(runIdsWithGroup):
    groupIdx, groupName, runId = runId.split(":")
    groupIdx = int(groupIdx) - 1
    runIds.append(runId)
    if groupName not in groupLabels:
      groupLabels[groupIdx] = groupName
      groups[groupIdx] = []
    groups[groupIdx].append(fileIdx)
  return runIds, groups, groupLabels, peptideQuantRows

def getPeptideQuantFileHeaders(peptideQuantFile):
  reader = csv.reader(open(peptideQuantFile, 'r'), delimiter = '\t')
  header = next(reader)
  return header

def parsePercPsmsPseudoPeptideFile(percPsmsFile):
  pseudoPeptRealPeptMap = defaultdict(list)
  for psm in percolator.parsePsmsPout(percPsmsFile):
    pseudoPept = c.clusterIdxToPseudoSeq(psm.scannr)
    pseudoPeptRealPeptMap[pseudoPept].append(psm._replace(id = pseudoPept))
  return pseudoPeptRealPeptMap

#########################################
## peptide1;linkPEP1,peptide2;linkPEP2 ##
#########################################

def parsePeptideLinkPEPs(peptideString):
  plps = dict()
  if len(peptideString) > 0:
    for x in peptideString.split(","):
      if ";" in x:
        peptide, linkPEP = x.split(";")
        linkPEP = float(linkPEP)
      else:
        peptide, linkPEP = x, 0.0
      plps[peptide] = linkPEP
  return plps

def serializePeptideLinkPEPs(peptideLinkPEPPairs):
  return ",".join([x + ";" + str(round(y,4)) for x,y in peptideLinkPEPPairs.items()])

def getQuantMatrix(quantRows, condenseChargeStates = True, retainBestChargeState = True):
  quantRows = list(quantRows) # contains full information about the quantification row
  if condenseChargeStates:
    peptQuantRowGroups = itertools.groupby(sorted(quantRows, key = lambda x : x.peptide[2:-2].replace("[UNIMOD:4]", "")), key = lambda x : getMods(x.peptide)[1])
    quantRows = list() # contains full information about the quantification row
    quantMatrix = list() # contains just the quantification values
    for peptSeq, pqrs in peptQuantRowGroups:
      pqrs = list(pqrs)
      if retainBestChargeState:
        pqrs = sorted(pqrs, key = lambda x : x.combinedPEP)
        quantMatrix.append([x if x > 0.0 else np.nan for x in pqrs[0].quant])
        quantRows.append(pqrs[0])
      else:
        quantMatrix.append([x if x > 0.0 else np.nan for x in pqrs[0].quant])
        if len(pqrs) > 1:
          peptQuantMatrix = [x.quant for x in pqrs]
          geomAvgQuant = list()
          for i in range(len(quantMatrix[-1])):
            quantsRun = [x[i] if x[i] > 0.0 else np.nan for x in peptQuantMatrix]
            if len(quantsRun) > 0:
              geomAvgQuant.append(weightedGeomAvg(quantsRun, [x.combinedPEP for x in pqrs]))
            else:
              geomAvgQuant.append(0.0)
          quantMatrix[-1] = geomAvgQuant
        quantRows.append(pqrs[0]._replace(qval = geomAvg([x.combinedPEP for x in pqrs])))
  else:
    quantMatrix = [[y if y > 0.0 else np.nan for y in x.quant] for x in quantRows]

  quantRows, quantMatrix = zip(*[(x, np.array(y)) for x, y in sorted(zip(quantRows, quantMatrix), key = lambda x : x[0].combinedPEP)])
  quantRows = list(quantRows)
  quantMatrix = list(quantMatrix)
  return quantRows, quantMatrix

def weightedGeomAvg(row, weights):
  weightSum = np.nansum(weights)
  if weightSum > 0:
    return np.exp(np.nansum(np.log(row) * weights) / weightSum)
  else:
    return np.nan

def geomAvg(row):
  return np.exp(np.nanmean(np.log(row)))

def geoNormalize(row):
  return row / geomAvg(row)
  
def getMods(modPeptide):
  peptide = ""
  peptideIdx = 0
  mods = [0]
  inMod = False
  for i in range(len(modPeptide)):
    if modPeptide[i] == "[":
      j = modPeptide[i:].find("]")
      if modPeptide[i+1:i+j].startswith("UNIMOD:"):
        unimodId = int(modPeptide[i+1:i+j].split(":")[1])
        if unimodId == 4: # Carboxyamidomethylation peptide N-term or aa
          if i == 0 or modPeptide[i-1] != "C":
            massDiff = 57.021464
          else:
            massDiff = 0.0
        elif unimodId == 5: # Carbamylation peptide N-term
          massDiff = 43.005814
        elif unimodId == 1: # Acetylation peptide N-term
          massDiff = 42.010565
        elif unimodId == 28 or unimodId == 385: # Pyro-glu from Q / Pyro-carbamidomethyl as a delta from Carbamidomethyl-Cys
          massDiff = -17.026549
        elif unimodId == 27: # Pyro-glu from E
          massDiff = -18.010565
        elif unimodId == 35: # Oxidation of M
          massDiff = 15.994915
        else:
          sys.exit("Unknown UNIMOD id: " + str(unimodId))
        mods[peptideIdx] += massDiff
      else:
        mods[peptideIdx] += float(modPeptide[i+1:i+j])
      if mods[peptideIdx] == 16.0: # more accurate monoisotopic mass for oxidations
        mods[peptideIdx] = 15.994915
      inMod = True
    elif modPeptide[i] == "]":
      inMod = False
    elif not inMod:
      peptide += modPeptide[i]
      peptideIdx += 1
      mods.append(0)
  return mods, peptide
