'''
Create Triqler input files by converting the evidence.txt output file from
MaxQuant.
'''

from __future__ import print_function

import sys
import os
import collections

import numpy as np

from .. import parsers
from ..triqler import __version__, __copyright__
from . import helpers

def main():
  print('Triqler.convert.maxquant version %s\n%s' % (__version__, __copyright__))
  print('Issued command:', os.path.basename(__file__) + " " + " ".join(map(str, sys.argv[1:])))
  
  args, params = parseArgs()
  
  convertMqToTriqler(args.file_list_file, args.in_file, args.out_file, params)

def parseArgs():
  import argparse
  apars = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  requiredNamed = apars.add_argument_group('required arguments')
  
  apars.add_argument('in_file', default=None, metavar = "IN_FILE",
                     help='''MaxQuant evidence.txt file.
                          ''')
  
  requiredNamed.add_argument('--file_list_file', metavar='L', 
                     help='Simple text file with spectrum file names in first column and condition in second column.',
                     required = True)
  
  apars.add_argument('--out_file', default = "triqler_input.tsv", metavar='OUT', 
                     help='''Path to triqler input file (writing in TSV format).
                          ''')
  
  apars.add_argument('--skip_normalization',
                     help='Skip retention-time based intensity normalization.',
                     action='store_true')
  
  apars.add_argument('--skip_mbr_rows',
                     help='Skips the match-between-runs rows in the output.',
                     action='store_true')
  
  apars.add_argument('--use_gene_names',
                     help='Use gene names instead of protein IDs as identifiers from the MaxQuant output.',
                     action='store_true')
  
  # ------------------------------------------------
  args = apars.parse_args()
  
  params = dict()
  params['simpleOutputFormat'] = True
  params['skipNormalization'] = args.skip_normalization
  params['skipMBR'] =  args.skip_mbr_rows
  params['useGeneNames'] =  args.use_gene_names
  params['plotScatter'] = False
  
  return args, params
  
def convertMqToTriqler(fileListFile, mqEvidenceFile, triqlerInputFile, params):
  fileInfoList = parsers.parseFileList(fileListFile)
  
  peptideToFeatureMap = parseMqEvidenceFile(mqEvidenceFile, fileInfoList, params)
  
  rTimeArrays, factorArrays = helpers.getNormalizationFactorArrays(peptideToFeatureMap, fileInfoList, params)
  
  helpers.writeTriqlerInputFile(triqlerInputFile, peptideToFeatureMap, rTimeArrays, factorArrays, params)

def parseMqEvidenceFile(mqEvidenceFile, fileInfoList, params):
  fileList, _, _, _ = zip(*fileInfoList)
  reader = parsers.getTsvReader(mqEvidenceFile)
  headers = next(reader) # save the header
  headers = list(map(str.lower, headers))
  
  peptCol = headers.index('modified sequence')
  fileCol = headers.index('raw file')
  chargeCol = headers.index('charge')
  intensityCol = headers.index('intensity')
  if params['useGeneNames']:
    geneCol = headers.index('gene names')
    fastaCol = headers.index('fasta headers')
    reverseCol = headers.index('reverse') 
    contaminantCol = headers.index('contaminant') 
  proteinCol = headers.index('leading proteins')
  scoreCol = headers.index('score')
  rtCol = headers.index('retention time')
  
  fractionCol = headers.index('fraction') if 'fraction' in headers else -1
  experimentCol = headers.index('experiment') if 'experiment' in headers else -1
  
  print("Parsing MaxQuant evidence.txt file")
  peptideToFeatureMap = collections.defaultdict(list)
  for lineIdx, row in enumerate(reader):
    if lineIdx % 500000 == 0:
      print("  Reading line", lineIdx)
    
    if params['useGeneNames']:
      if row[reverseCol] == "+":
        proteins = getGeneNames(row[fastaCol], "REV__")
      elif row[contaminantCol] == "+":
        proteins = getGeneNames(row[fastaCol], "CON__")
      else:
        proteins = row[geneCol].split(";")
      
      if len(proteins) == 0:
        proteins = row[proteinCol].split(";")
      else:
        proteins = list(set(proteins))
    else:
      proteins = row[proteinCol].split(";")
    
    linkPEP = 0.0
    key = (row[peptCol], row[chargeCol])
    
    if not row[fileCol] in fileList:
      print("Warning: Could not find %s in the specified file list, skipping row" % row[fileCol])
      continue
    
    fileIdx = fileList.index(row[fileCol])
    run, condition, sample, fraction = fileInfoList[fileIdx]
    if fraction == -1 and fractionCol != -1:
      sample, fraction = row[experimentCol], row[fractionCol]
    
    if key in peptideToFeatureMap:
      featureClusterIdx = peptideToFeatureMap[key][0][0].featureClusterId
    else:
      featureClusterIdx = len(peptideToFeatureMap)
    
    if row[intensityCol] == "" or float(row[scoreCol]) <= 0:
      continue
    
    triqlerRow = parsers.TriqlerInputRow(sample, condition, row[chargeCol], lineIdx, linkPEP, featureClusterIdx, np.log(float(row[scoreCol])), float(row[intensityCol]), row[peptCol], proteins)
    peptideToFeatureMap[key].append((triqlerRow, float(row[rtCol]), fraction))
  
  return peptideToFeatureMap

def getGeneNames(fastaHeaders, prefix):
  fastaHeaders = fastaHeaders.split(";")
  proteins = []
  for h in fastaHeaders:
    if "GN=" in h:
      proteins.append(prefix + h.split("GN=")[1].split()[0])
  return proteins

if __name__ == "__main__":
   main()
