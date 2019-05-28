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
import itertools
import re
from collections import defaultdict, namedtuple

################################################
## input: filename <tab> group (one per line) ##
################################################

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

def getRunIds(params):
  return [getGroupLabel(idx, params['groups'], params['groupLabels']) + ":" + x.split("/")[-1] for idx, x in enumerate(params['fileList'])]

############################
## Feature cluster files  ##
############################

PrecursorCandidate = namedtuple("PrecursorCandidate", "fileIdx fileName precMz charge rTime intensity peptLinkPEPs")

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

##########################
## Triqler input files  ##
##########################

TriqlerSimpleInputRowHeaders = "run condition charge searchScore intensity peptide proteins".split(" ")
TriqlerInputRowHeaders = "run condition charge spectrumId linkPEP featureClusterId searchScore intensity peptide proteins".split(" ")
TriqlerInputRowBase = namedtuple("TriqlerInputRow", TriqlerInputRowHeaders)

class TriqlerInputRow(TriqlerInputRowBase):
  def toList(self):
    l = list(self)
    return l[:-1] + l[-1]
  
  def toSimpleList(self):
    l = list(self)
    return l[:3] + l[6:-1] + l[-1]
  
  def toString(self):
    return "\t".join(map(str, self.toList()))

def parseTriqlerInputFile(triqlerInputFile):
  reader = csv.reader(open(triqlerInputFile, 'r'), delimiter = '\t')
  headers = next(reader)
  hasLinkPEPs = "linkPEP" in headers
  getUniqueProteins = lambda x : list(set([p for p in x if len(p.strip()) > 0]))
  intensityCol = 7 if hasLinkPEPs else 4
  seenPeptChargePairs = dict()
  for i, row in enumerate(reader):
    if i % 1000000 == 0:
      print("  Reading row", i)
    
    intensity = float(row[intensityCol])
    if intensity > 0.0:
      if hasLinkPEPs:
        proteins = getUniqueProteins(row[9:])
        yield TriqlerInputRow(row[0], row[1], int(row[2]), int(row[3]), float(row[4]), int(row[5]), float(row[6]), intensity, row[8], proteins)
      else:
        key = (int(row[2]), row[5])
        if key not in seenPeptChargePairs:
          seenPeptChargePairs[key] = len(seenPeptChargePairs)
        proteins = getUniqueProteins(row[6:])
        yield TriqlerInputRow(row[0], row[1], int(row[2]), (i+1) * 100, 0.0, seenPeptChargePairs[key], float(row[3]), intensity, row[5], proteins)

def hasLinkPEPs(triqlerInputFile):
  reader = csv.reader(open(triqlerInputFile, 'r'), delimiter = '\t')
  headers = next(reader)
  return "linkPEP" in headers

##############################
## Peptide quant row files  ##
##############################

PeptideQuantRowHeaders = "combinedPEP charge featureGroup spectrum linkPEP quant identificationPEP peptide protein".split(" ")
PeptideQuantRowBase = namedtuple("PeptideQuantRow", PeptideQuantRowHeaders)

class PeptideQuantRow(PeptideQuantRowBase):
  def toList(self):
    l = list(self)
    return l[:4] + list(map(lambda x : '%.5g' % x, l[4])) + list(map(lambda x : '%.2f' % x, l[5])) + list(map(lambda x : '%.5g' % x, l[6])) + l[7:8] + [";".join(map(lambda x : x.replace(";", "_"), l[8]))]

  def toString(self):
    return "\t".join(map(str, self.toList()))

def getPeptideQuantRowHeaders(runs):
  return PeptideQuantRowHeaders[:4] + runs + runs + runs + PeptideQuantRowHeaders[7:]

def parsePeptideQuantFile(peptideQuantFile):
  reader = csv.reader(open(peptideQuantFile, 'r'), delimiter = '\t')
  header = next(reader)
  numRuns = (header.index("peptide") - header.index("spectrum") - 1) / 3
  peptideQuantRows = list()
  for row in reader:
    if len(row) > 6+3*numRuns:
      proteins = row[5+3*numRuns:]
    else:
      proteins = row[5+3*numRuns].split(";")
    peptideQuantRows.append(PeptideQuantRow(float(row[0]), int(row[1]), int(row[2]), int(row[3]), np.array(map(float, row[4:4+numRuns])), np.array(map(float, row[4+numRuns:4+2*numRuns])), np.array(map(float, row[4+2*numRuns:4+3*numRuns])), row[4+3*numRuns], proteins))

  runIdsWithGroup = header[4:4+numRuns]
  maxGroups = len(set([runId.split(":")[0] for runId in runIdsWithGroup]))
  runIds = list()
  groups, groupLabels = [None] * maxGroups, [None] * maxGroups
  groupNames = list()
  for fileIdx, runId in enumerate(runIdsWithGroup):
    runIdSplitted = runId.split(":")
    if len(runIdSplitted) == 3:
      groupIdx, groupName, runId = runIdSplitted
      groupIdx = int(groupIdx) - 1
    else:
      groupName, runId = runIdSplitted
      if groupName not in groupNames:
        groupNames.append(groupName)
      groupIdx = groupNames.index(groupName)
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

def filterAndGroupPeptides(peptQuantRows, peptFilter = lambda x : True):
  validPqr = lambda x : len(x.protein) == 1 and x.protein[0] != "NA" and x.combinedPEP < 1.0
  peptQuantRows = filter(lambda x : validPqr(x) and peptFilter(x), peptQuantRows)
  protQuantRows = itertools.groupby(sorted(peptQuantRows, key = lambda x : x.protein[0]), key = lambda x : x.protein[0])
  return protQuantRows

###################################
## Quant matrix helper functions ##
###################################

def getQuantMatrix(quantRows, condenseChargeStates = True, retainBestChargeState = True):
  quantRows = list(quantRows) # contains full information about the quantification row
  if condenseChargeStates:
    peptQuantRowGroups = itertools.groupby(sorted(quantRows, key = lambda x : cleanPeptide(x.peptide)), key = lambda x : cleanPeptide(x.peptide))
    quantRows = list() # contains full information about the quantification row
    quantMatrix = list() # contains just the quantification values
    for _, pqrs in peptQuantRowGroups:
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

def cleanPeptide(peptide):
  if peptide[1] == "." and peptide[-2] == ".":
    peptide = peptide[2:-2]
  return re.sub('\[[-0-9]*\]', '', peptide)
