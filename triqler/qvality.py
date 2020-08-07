# This is a python port of the C++ code of qvality (https://github.com/percolator/percolator/)
# It does not include the mix-max corrections, nor the pi0 corrections

from __future__ import print_function

import subprocess
import tempfile
import csv
import os
import sys

import numpy as np
import bisect

tao = 2.0 / (1 + np.sqrt(5.0)) # inverse of golden section
scaleAlpha = 1
stepEpsilon = 1e-8
gRange = 35.0
weightSlope = 1e1

VERB = 3

# pi0 estimation parameters
numLambda = 100
maxLambda = 0.5

# this function returns PEPs in ascending order (lowest PEP first)
def getQvaluesFromScores(targetScores, decoyScores, includePEPs = False, includeDecoys = False, tdcInput = False, pi0 = 1.0, plotRegressionCurve = False, numBins = 500):
  if type(targetScores) is not np.ndarray:
    targetScores = np.array(targetScores)
  if type(decoyScores) is not np.ndarray:
    decoyScores = np.array(decoyScores)
  
  if len(targetScores) == 0:
    sys.exit("ERROR: no target hits available for PEP calculation")
  
  if len(decoyScores) == 0:
    sys.exit("ERROR: no decoy hits available for PEP calculation")
  
  targetScores.sort()
  decoyScores.sort()
  allScores = np.concatenate((targetScores, decoyScores))
  allScores.sort()
  
  medians, negatives, sizes = binData(allScores, decoyScores, numBins)
  medians, negatives, sizes = np.array(medians), np.array(negatives), np.array(sizes)
  
  # sort in descending order, highest score first
  if includeDecoys:
    evalScores = allScores[::-1]
    #evalScores = np.array([x[0] for x in combined])
  else:
    evalScores = targetScores[::-1]
  
  if VERB > 3:
    print(medians, negatives, sizes)
  
  variables = roughnessPenaltyIRLS(medians, negatives, sizes)
  
  if pi0 < 1.0:
    factor = pi0 * float(len(targetScores)) / len(decoyScores)
  else:
    factor = 1.0
  
  if plotRegressionCurve:
    scoresForPlot = evalScores.copy()
    
  probs = factor * np.exp(splineEval(evalScores, medians, variables))
  probs = monotonize(probs)
  
  if plotRegressionCurve:
    import matplotlib.pyplot as plt
    plt.plot(medians, (1.0*negatives) / sizes, '*-')
    plt.plot(scoresForPlot, probs)
    
    plt.figure()
    plt.plot(medians, (1.0*negatives) / sizes, '*-')
    plt.plot(scoresForPlot, probs)
    plt.yscale("log")
    plt.show()
  return None, probs
  
def getQvaluesFromPvalues(pvalues, includePEPs = False):
  targetScores = sorted(pvalues)
  pi0 = estimatePi0(targetScores)
  if VERB > 2:
    print("Estimating pi0 = %f" % pi0)
  
  step = 1.0 / 2.0 / len(pvalues)
  decoyScores = np.arange(step, 1 - step + 1e-10, step*2)

  targetScores = pvaluesToScores(targetScores)
  decoyScores = pvaluesToScores(decoyScores)
  return getQvaluesFromScores(targetScores, decoyScores, includePEPs, includeDecoys = False, pi0 = pi0)

def pvaluesToScores(pvalues):
  return np.array(list(map(lambda x : -1*np.log(x / (1 - x)), pvalues)))
  
def monotonize(peps):
  return np.minimum(1.0, np.maximum.accumulate(peps))

def binData(allScores, decoyScores, numBins):
  binEdges = list(map(lambda x : int(np.floor(x)), np.linspace(0, len(allScores), numBins+1)))
  bins = list()
  startIdx = 0
  for endIdx in binEdges[1:]:
    if startIdx < endIdx:
      while endIdx < len(allScores) and allScores[endIdx-1] == allScores[endIdx]:
        endIdx += 1
      bins.append(allScores[startIdx:endIdx])
      startIdx = endIdx
  
  results = list()
  for b in bins:
    m = np.median(b)
    numNegs = np.searchsorted(decoyScores, b[-1], side = 'right') - np.searchsorted(decoyScores, b[0], side = 'left')
    numTot = len(b)
    results.append([m, numNegs, numTot])
  return zip(*results)

def roughnessPenaltyIRLS(medians, negatives, sizes):
  Q, R = initQR(medians)
  g, w, z, gamma, p, gnew = initg(negatives, sizes)
  variables = (Q, R, g, w, z, gamma, p, gnew)
  
  p1 = 1.0 - tao
  p2 = tao
  
  cv1 = evaluateSlope(medians, negatives, sizes, variables, -scaleAlpha * np.log(p1))
  cv2 = evaluateSlope(medians, negatives, sizes, variables, -scaleAlpha * np.log(p2))
  
  alpha = alphaLinearSearchBA(0.0, 1.0, p1, p2, cv1, cv2, medians, negatives, sizes, variables)
  if VERB > 3:
    print("Alpha selected to be", alpha)
  variables = iterativeReweightedLeastSquares(medians, negatives, sizes, variables, alpha)
  return variables

def alphaLinearSearchBA(min_p, max_p, p1, p2, cv1, cv2, medians, negatives, sizes, variables):
  # Minimize Slope score
  # Use neg log of 0<p<1 so that we allow for searches 0<alpha<inf
  oldCV = 0.0
  if cv2 < cv1:
    # keep point 2
    min_p = p1
    p1 = p2
    p2 = min_p + tao * (max_p - min_p)
    oldCV = cv1
    cv1 = cv2
    cv2 = evaluateSlope(medians, negatives, sizes, variables, -1*scaleAlpha*np.log(p2))
    if VERB > 3:
      print("New point with alpha=", -scaleAlpha*np.log(p2), ", giving slopeScore=", cv2)
  else:
    # keep point 1
    max_p = p2
    p2 = p1
    p1 = min_p + (1 - tao) * (max_p - min_p)
    oldCV = cv2
    cv2 = cv1
    cv1 = evaluateSlope(medians, negatives, sizes, variables, -1*scaleAlpha*np.log(p1))
    if VERB > 3:
      print("New point with alpha=", -scaleAlpha*np.log(p1), ", giving slopeScore=", cv1)
  if (oldCV - min(cv1, cv2)) / oldCV < 1e-5 or abs(p2 - p1) < 1e-10:
    return -scaleAlpha*np.log(p1) if cv1 < cv2 else -scaleAlpha*np.log(p2)
  return alphaLinearSearchBA(min_p, max_p, p1, p2, cv1, cv2, medians, negatives, sizes, variables)

def evaluateSlope(medians, negatives, sizes, variables, alpha):
  # Calculate a spline for current alpha
  variables = iterativeReweightedLeastSquares(medians, negatives, sizes, variables, alpha)
  
  _, _, g, _, _, _, _, _ = variables
  
  # Find highest point (we only want to evaluate things to the right of that point)
  n = len(medians)
  mixg = 1 # Ignore 0 and n-1
  maxg = g[mixg]
  for ix in range(mixg, n-1):
    if g[ix] >= maxg:
      maxg = g[ix]
      mixg = ix
  maxSlope = -10e6
  slopeix = -1
  for ix in range(mixg+1, n-2):
    slope = g[ix-1]-g[ix]
    if slope > maxSlope:
      maxSlope = slope
      slopeix = ix
  
  # Now score the fit based on a linear combination between
  # The bump area and alpha
  if VERB > 3:
    print("mixg=", mixg, ", maxg=", maxg, ", maxBA=", maxSlope, " at ix=", slopeix, ", alpha=", alpha)
  
  return maxSlope * weightSlope + alpha

def iterativeReweightedLeastSquares(medians, negatives, sizes, variables, alpha, epsilon = stepEpsilon, maxiter = 50):
  Q, R, g, w, z, gamma, p, gnew = variables
  for it in range(maxiter):
    g = gnew
    p, z, w = calcPZW(g, negatives, sizes)
    aWiQ = np.multiply((alpha / w)[:,None], Q)
    M = R + Q.T.dot(aWiQ)
    gamma = np.linalg.solve(M, Q.T.dot(z))
    gnew = z - aWiQ.dot(gamma)
    gnew = np.minimum(gRange, np.maximum(-1*gRange, gnew))
    difference = g - gnew
    step = np.linalg.norm(difference) / len(medians)
    if VERB > 3:
      print("Step size:", step)
    if step < epsilon:
      return (Q, R, g, w, z, gamma, p, gnew)
  
  if VERB > 1:
    print("Warning: IRLS did not converge with maxIter =", maxiter)
  
  return (Q, R, g, w, z, gamma, p, gnew)

def calcPZW(g, negatives, sizes, epsilon = 1e-15):
  e = np.exp(g)
  p = np.minimum(1 - epsilon, np.maximum(epsilon, e / (1+e)))
  w = np.maximum(epsilon, sizes * p * (1 - p))
  z = np.minimum(gRange, np.maximum(-1*gRange, g + (negatives - p * sizes) / w))
  return p, z, w

def initg(negatives, sizes):
  n = len(negatives)
  g = np.zeros(n)
  w = np.zeros(n)
  z = np.ones(n) * 0.5
  gamma = np.zeros(n-2)
  
  p = (negatives + 0.05) / (sizes + 0.1)
  gnew = np.log(p / (1-p))
  return g, w, z, gamma, p, gnew
  
def initQR(medians):
  n = len(medians)
  dx = medians[1:] - medians[:-1]
  Q = np.zeros((n, n -2))
  Q[range(n-2), range(n-2)] = 1.0 / dx[:-1]
  Q[range(1,n-1), range(n-2)] = - 1.0 / dx[:-1] - 1.0 / dx[1:]
  Q[range(2,n), range(n-2)] = 1.0 / dx[1:]
  
  R = np.zeros((n-2, n-2))
  R[range(n-2), range(n-2)] = (dx[:-1] + dx[1:]) / 3
  R[range(n-3), range(1,n-2)] = dx[1:-1] / 6
  R[range(1,n-2), range(n-3)] = dx[1:-1] / 6
  return Q, R

def splineEval(scores, medians, variables):
  _, _, g, _, _, gamma, _, _ = variables
  #score = np.exp(score)
  n = len(medians)
  rights = np.searchsorted(medians, scores)
  
  derr = (g[1] - g[0]) / (medians[1] - medians[0]) - (medians[1] - medians[0]) / 6 * gamma[0]
  scores[rights == 0] = g[0] - (medians[0] - scores[rights == 0]) * derr # reuse "scores" array to save memory
  
  derl = (g[-1] - g[-2]) / (medians[-1] - medians[-2]) + (medians[-1] - medians[-2]) / 6 * gamma[-3]
  scores[rights == n] = g[-1] + (scores[rights == n] - medians[-1]) * derl
  
  idxs = np.where((rights > 0) & (rights < n))
  rights = rights[idxs] # reuse "rights" array to save memory
  hs = medians[rights] - medians[rights - 1]
  
  drs = medians[rights] - scores[idxs]
  dls = scores[idxs] - medians[rights - 1]
  
  gamr = np.zeros_like(hs)
  gamr[rights < (n - 1)] = gamma[rights[rights < (n - 1)] - 1]
  
  gaml = np.zeros_like(hs)
  gaml[rights > 1] = gamma[rights[rights > 1] - 2]
  
  scores[idxs] = (dls * g[rights] + drs * g[rights - 1]) / hs - dls * drs / 6 * ((1.0 + dls / hs) * gamr + (1.0 + drs / hs) * gaml)
  
  return scores

def estimatePi0(pvalues, numBoot = 100):
  pvalues = np.array(pvalues)
  lambdas, pi0s = list(), list()
  numPvals = len(pvalues)
  
  for lambdaIdx in range(numLambda + 1):
    l = ((lambdaIdx + 1.0) /  numLambda) * maxLambda
    startIdx = np.searchsorted(pvalues, l)
    Wl = numPvals - startIdx
    pi0 = Wl / (1.0 - l) / numPvals
    if pi0 > 0.0:
      lambdas.append(l)
      pi0s.append(pi0)
  
  if len(pi0s) == 0:
    print("Error in the input data: too good separation between target and decoy PSMs.\nImpossible to estimate pi0, setting pi0 = 1")
    return 1.0
  
  minPi0 = min(pi0s)
  
  mse = [0.0] * len(pi0s)
  # Examine which lambda level is most stable under bootstrap
  for boot in range(numBoot):
    pBoot = bootstrap(pvalues)
    n = len(pBoot)
    for idx, l in enumerate(lambdas):
      startIdx = np.searchsorted(pvalues, l)
      Wl = numPvals - startIdx
      pi0Boot = Wl / n / (1 - l)
      # Estimated mean-squared error.
      mse[idx] += (pi0Boot - minPi0) * (pi0Boot - minPi0)
  return max([min([pi0s[np.argmin(mse)], 1.0]), 0.0])

def bootstrap(allVals, maxSize = 1000):
  return sorted(np.random.choice(allVals, min([len(allVals), maxSize]), replace = False))
  
def getQvaluesFromPvaluesQvality(pvalues, includePEPs = False):
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
    
def getQvaluesFromScoresQvality(targetScores, decoyScores, includePEPs = False, includeDecoys = False, tdcInput = False):
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
  
  rc = subprocess.call("qvality %s %s %s -v %d > %s" % (tdcInputFlag, targetFile, decoyFile, VERB, qvalFile), shell=True)
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

def countBelowFDR(peps, qvalThreshold):
  return np.count_nonzero(np.cumsum(peps) / np.arange(1, len(peps) + 1) < qvalThreshold)

def getPEPFromScoreLambda(targetScores, decoyScores):
  if len(decoyScores) == 0:
    sys.exit("ERROR: No decoy hits found, check if the correct decoy prefix was specified with the --decoy_pattern flag")
  
  targetScores = np.array(targetScores)
  decoyScores = np.array(decoyScores)
  _, peps = getQvaluesFromScores(targetScores, decoyScores, includePEPs = True, includeDecoys = True, tdcInput = True)
  
  print("  Identified", countBelowFDR(peps, 0.01), "PSMs at 1% FDR")
  
  peps = peps[::-1] # PEPs in descending order, highest PEP first
  
  allScores = np.concatenate((targetScores, decoyScores))
  allScores.sort()  # scores in ascending order, lowest score first
  getPEPFromScore = lambda score : peps[min(np.searchsorted(allScores, score, side = 'left'), len(peps) - 1)] if not np.isnan(score) else 1.0
  
  return getPEPFromScore

def fdrsToQvals(fdrs):
  qvals = [0] * len(fdrs)
  if len(fdrs) > 0:
    qvals[len(fdrs)-1] = fdrs[-1]
    for i in range(len(fdrs)-2, -1, -1):
      qvals[i] = min(qvals[i+1], fdrs[i])
  return qvals

def getPEPAtFDRThreshold(peps, qvalThreshold):
  qvals = np.divide(np.cumsum(peps), np.arange(1,len(peps)+1))
  if qvals[-1] > qvalThreshold:
    idx = np.argmax(qvals > qvalThreshold)
  else:
    idx = -1
  pepThreshold = peps[idx]
  return pepThreshold
   
def unitTestScoreInput():
  import scipy.stats
  import matplotlib.pyplot as plt
  import time
  
  for s in range(3,4):
    t0 = time.time()
    np.random.seed(s)
    N = 500000
    #targetScores = np.round(np.random.normal(2.0,1,N) + np.random.normal(0.0,1,N), 1)
    #decoyScores = np.round(np.random.normal(0.0,1,N), 1)
    targetScores = np.random.normal(2.0,1,N) + np.random.normal(0.0,1,N)
    decoyScores = np.random.normal(0.0,1,N)
    print("Generated input", time.time() - t0, "sec")
    
    t0 = time.time()
    _, peps1 = getQvaluesFromScores(targetScores, decoyScores, includePEPs = True, includeDecoys = True, tdcInput = True, plotRegressionCurve = False)
    print("Python", time.time() - t0, "sec")
    
    t0 = time.time()
    _, peps2 = getQvaluesFromScoresQvality(targetScores, decoyScores, includePEPs = True, includeDecoys = True, tdcInput = True)
    print("C++", time.time() - t0, "sec")
    
    sortedScores = sorted(targetScores + decoyScores, reverse = True)
    plt.figure(s+1)
    plt.plot(scipy.stats.norm.pdf(sortedScores, 0.0,1) / (scipy.stats.norm.pdf(sortedScores, 2.0,1) + scipy.stats.norm.pdf(sortedScores, 0.0,1)), label = 'Ground truth')
    plt.plot(peps1, label = 'Python')
    plt.plot(peps2, label = 'C++')
    plt.legend()
  plt.show()

def unitTestPvalInput():
  import scipy.stats
  import matplotlib.pyplot as plt
  
  for s in range(3,4):
    np.random.seed(s)
    N = 5000
    targetScores = np.round(np.concatenate((np.random.uniform(0,1,N), scipy.stats.truncnorm.rvs(0.0,1,loc=0,scale=0.5,size=N))), 8)
    _, peps1 = getQvaluesFromPvalues(targetScores, includePEPs = True)
    _, peps2 = getQvaluesFromPvaluesQvality(targetScores, includePEPs = True)
    
    plt.figure(s+1)
    targetScores = sorted(targetScores)
    plt.plot(scipy.stats.uniform.pdf(targetScores, 0,1) / (scipy.stats.truncnorm.pdf(targetScores, 0.0,1,loc=0,scale=0.5) + scipy.stats.uniform.pdf(targetScores, 0,1)), label = 'Ground truth')
    plt.plot(peps1, label = 'Python')
    plt.plot(peps2, label = 'C++')
    plt.legend()
  plt.show()

if __name__ == "__main__":
  unitTestScoreInput()
  unitTestPvalInput()
