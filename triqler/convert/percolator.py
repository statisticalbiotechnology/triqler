#!/usr/bin/python

'''
This script contains helper functions to parse percolator in and output files (tab delimited formats)
'''

from __future__ import print_function

import sys
import csv
import collections

from .. import parsers

PercolatorPoutPsmsBase = collections.namedtuple("PercolatorPoutPsms", "id filename scannr charge svm_score qvalue PEP peptide proteins")

class PercolatorPoutPsms(PercolatorPoutPsmsBase):
  def toList(self):
    return [self.id, self.svm_score, self.qvalue, self.PEP, self.peptide] + self.proteins
  
  def toString(self):
    return "\t".join(map(str, self.toList()))
  
# works for peptide and psms pouts
def parsePsmsPout(poutFile, qThresh = 1.0, proteinMap = None, parseId = True, fixScannr = False):
  reader = parsers.getTsvReader(poutFile)
  headers = next(reader) # save the header
  
  cruxOutput = True if "percolator score" in headers else False
  if cruxOutput:
    proteinCol = headers.index('protein id')
    fileIdxCol = headers.index('file_idx')
    scanCol = headers.index('scan')
    chargeCol = headers.index('charge')
    scoreCol = headers.index('percolator score')
    qvalCol = headers.index('percolator q-value')
    postErrCol = headers.index('percolator PEP')
    peptideCol = headers.index('sequence')
    terminalsCol = headers.index('flanking aa')
  else:
    proteinCol = headers.index('proteinIds')
    qvalCol = headers.index('q-value')
  
  fixScannr = "_msfragger" in poutFile or "_moda" in poutFile or "_msgf" in poutFile or fixScannr
  for row in reader:
    if float(row[qvalCol]) <= qThresh:
      if cruxOutput:
        proteins = list(set(row[proteinCol].split(',')))
      else:
        proteins = row[5:]
      
      if proteinMap:
        proteins = list(map(proteinMap, proteins))
      
      if cruxOutput:
        yield PercolatorPoutPsms(row[fileIdxCol] + "_" + row[scanCol] + "_" + row[chargeCol], int(row[fileIdxCol]), int(row[scanCol]), int(row[chargeCol]), float(row[scoreCol]), float(row[qvalCol]), float(row[postErrCol]), row[terminalsCol][0] + "." + row[peptideCol] + "." + row[terminalsCol][1], proteins)
      elif parseId:
        yield PercolatorPoutPsms(row[0], getFileName(row[0], fixScannr), getId(row[0], fixScannr), getCharge(row[0]), float(row[1]), float(row[qvalCol]), float(row[3]), row[4], proteins)
      else:
        yield PercolatorPoutPsms(row[0], "", 0, 0, float(row[1]), float(row[2]), float(row[3]), row[4], proteins)
    else:
      break

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

if __name__ == "__main__":
   main(sys.argv[1:])
