#!/usr/bin/python

'''
This script contains helper functions to parse percolator in and output files (tab delimited formats)
'''

from __future__ import print_function

import os
import sys
import csv
import collections
import subprocess
import tempfile

import numpy as np

headers_ = []
hasDefaultDirection_ = False
defaultDirection_ = []

psmsPoutHeaders = "PSMId score q-value posterior_error_prob peptide proteinIds".split(" ")

PercolatorPoutPsmsBase = collections.namedtuple("PercolatorPoutPsms", "id filename scannr charge svm_score qvalue PEP peptide proteins")

class PercolatorPoutPsms(PercolatorPoutPsmsBase):
  def toList(self):
    return [self.id, self.svm_score, self.qvalue, self.PEP, self.peptide] + self.proteins
  
  def toString(self):
    return "\t".join(map(str, self.toList()))
    
def getDefaultPoutPsms():
  return PercolatorPoutPsms("", "", 0, 0, 0, 1.0, 1.0, "NA", ["NA"])
  
# works for peptide and psms pouts
def parsePsmsPout(poutFile, qThresh = 1.0, proteinMap = None, parseId = True, fixScannr = False):
  reader = csv.reader(open(poutFile, 'r'), delimiter='\t')
  headers = next(reader) # save the header
  
  fixScannr = "_msfragger" in poutFile or "_moda" in poutFile or "_msgf" in poutFile or fixScannr
  for row in reader:
    if float(row[2]) <= qThresh:
      proteins = row[5:]
      if proteinMap:
        proteins = map(proteinMap, proteins)
      if parseId:
        yield PercolatorPoutPsms(row[0], getFileName(row[0], fixScannr), getId(row[0], fixScannr), getCharge(row[0]), float(row[1]), float(row[2]), float(row[3]), row[4], proteins)
      else:
        yield PercolatorPoutPsms(row[0], "", 0, 0, float(row[1]), float(row[2]), float(row[3]), row[4], proteins)
    else:
      break

def parsePin(pinFile):
  global headers_, hasDefaultDirection_, defaultDirection_
  reader = csv.reader(open(pinFile, 'r'), delimiter='\t')
  headers_ = next(reader) # save the header
  headers = map(lambda x : x.lower(), headers_)
  
  PercolatorFeatureRow = collections.namedtuple("PercolatorFeatureRow", headers)
  for row in reader:
    if row[0] != "DefaultDirection":
      yield PercolatorFeatureRow(*(row[:len(headers)-1] + [row[len(headers)-1:]]))
    else:
      hasDefaultDirection_ = True
      defaultDirection_ = row

def hasDefaultDirection():
  return hasDefaultDirection_

def toList(psm):
  l = list(psm)
  return l[:-1] + l[-1]
  
def getId(PSMid, msgf = False):
  if msgf:
    return int((PSMid.split('_'))[-3]) / 100
  else:
    return int((PSMid.split('_'))[-3])

def getCharge(PSMid):
  return int((PSMid.split('_'))[-2])

def getFileName(PSMid, msgf = False):
  if msgf:
    return '_'.join(PSMid.split('_')[:-6])
  else:
    return '_'.join(PSMid.split('_')[:-3])

def write(outputFile, filteredList, permissions = 'a'):
  writer = csv.writer(open(outputFile,permissions), delimiter='\t')
  writer.writerow(headers_)
  if hasDefaultDirection():
    writer.writerow(defaultDirection_)
  for row in filteredList:
    writer.writerow(row)
  
def getQvalues(pvalues, includePEPs = False):
  fdp, pvalFile = tempfile.mkstemp()
  with open(pvalFile, 'w') as w:
    for pval in pvalues:
      w.write(str(pval) + '\n')
  
  fdq, qvalFile = tempfile.mkstemp()
  rc = subprocess.call("qvality %s > %s" % (pvalFile, qvalFile), shell=True)
  if includePEPs:
    qvals, peps = parseQvalues(qvalFile, includePEPs = includePEPs)
  else:
    qvals = parseQvalues(qvalFile)
  
  os.unlink(pvalFile)
  os.unlink(qvalFile)
  if includePEPs:
    return qvals, peps
  else:
    return qvals

def getQvaluesFromScores(targetScores, decoyScores, includePEPs = False, includeDecoys = False, tdcInput = False):
  fdp, targetFile = tempfile.mkstemp()
  with open(targetFile, 'w') as w:
    for s in targetScores:
      w.write(str(s) + '\n')
  
  fdp2, decoyFile = tempfile.mkstemp()
  with open(decoyFile, 'w') as w:
    for s in decoyScores:
      w.write(str(s) + '\n')
  
  fdq, qvalFile = tempfile.mkstemp()
  if tdcInput:
    tdcInputFlag = "-Y"
  else:
    tdcInputFlag = ""
  
  rc = subprocess.call("qvality %s %s %s > %s" % (tdcInputFlag, targetFile, decoyFile, qvalFile), shell=True)
  if includePEPs:
    qvals, peps = parseQvalues(qvalFile, includePEPs = includePEPs)
  else:
    qvals = parseQvalues(qvalFile)
  
  if includeDecoys:
    targetScores = np.array(targetScores[::-1])
    maxIdx = len(targetScores) - 1
    for s in decoyScores[::-1]:
      idx = np.searchsorted(targetScores, s, side='left')
      qvals.append(qvals[maxIdx-idx])
      if includePEPs:
        peps.append(peps[maxIdx-idx])
    qvals = sorted(qvals)
    if includePEPs:
      peps = sorted(peps)
      
  os.unlink(targetFile)
  os.unlink(decoyFile)
  os.unlink(qvalFile)
  if includePEPs:
    return qvals, peps
  else:
    return qvals
    
def parseQvalues(qvalFile, includePEPs = False):
  reader = csv.reader(open(qvalFile, 'r'), delimiter = '\t')
  next(reader)
  
  qvals, peps = list(), list()
  for row in reader:
    if includePEPs:
      peps.append(float(row[1]))
    qvals.append(float(row[2]))
  
  if includePEPs:
    return qvals, peps
  else:
    return qvals

def runPercolator(percolatorInputFile, force = False, args = "", plotScoreHist = False):
  if "/pin/" in percolatorInputFile:
    outputFolder, baseName = os.path.split(percolatorInputFile)
    baseName = baseName.replace(".tab", "")
    outputFolder = "/".join(outputFolder.split("/")[:-1])
    if "-Y" in args:
      tabFolder = os.path.join(outputFolder, 'tab_tdc')
      stdoutFolder = os.path.join(outputFolder, 'stdout_tdc')
    else:
      tabFolder = os.path.join(outputFolder, 'tab_mixmax')
      stdoutFolder = os.path.join(outputFolder, 'stdout_mixmax')
    
    percolatorOutputFile = os.path.join(tabFolder, baseName + ".percolator.tab.psms")
    percolatorDecoyOutputFile = os.path.join(tabFolder, baseName + ".percolator.decoys.tab.psms")
    args += " -m %s" % percolatorOutputFile
    args += " -M %s" % percolatorDecoyOutputFile
    args += " -r %s" % os.path.join(tabFolder, baseName + ".percolator.tab.peptides")
    args += " -B %s" % os.path.join(tabFolder, baseName + ".percolator.decoys.tab.peptides")
    if "-f" in args:
      args += " -l %s" % os.path.join(tabFolder, baseName + ".percolator.tab.proteins")
      args += " -L %s" % os.path.join(tabFolder, baseName + ".percolator.decoys.tab.proteins")
    percolatorLogFile = os.path.join(stdoutFolder, baseName + ".stdout.txt")
  else:
    percolatorOutputFile = percolatorInputFile.replace(".pin","") + ".pout"
    percolatorDecoyOutputFile = percolatorInputFile.replace(".pin","") + ".decoys.pout"
    percolatorLogFile = percolatorInputFile.replace(".pin","") + ".stderr.log"
    args += " -r %s -B %s" % (percolatorOutputFile, percolatorDecoyOutputFile)
  
  if not os.path.isfile(percolatorOutputFile) or force:
    cmd = "percolator %s %s 2>&1 | tee %s" % (percolatorInputFile, args, percolatorLogFile)
    
    rc = subprocess.call(cmd, shell=True)
    if rc == 1:
      print("Error while processing " + cmd)
  
  return percolatorOutputFile, percolatorDecoyOutputFile

if __name__ == "__main__":
   main(sys.argv[1:])
