# This is a python port of the C++ code of qvality (https://github.com/percolator/percolator/)
# It does not include the mix-max corrections, nor the pi0 corrections

from __future__ import print_function

import subprocess
import tempfile
import csv
import os

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

def getQvaluesFromScores(targetScores, decoyScores, includePEPs = False, includeDecoys = False, tdcInput = False, pi0 = 1.0):
  combined = list(map(lambda x : (x, True), targetScores)) + list(map(lambda x : (x, False), decoyScores))
  np.random.shuffle(combined) # shuffle all scores so that target and decoy hits with identical scores are in a random order later
  combined = sorted(combined, reverse = True)
  
  medians, negatives, sizes = binData(combined)
  
  medians, negatives, sizes = np.array(medians[::-1]), np.array(negatives[::-1]), np.array(sizes[::-1])
  if VERB > 3:
    print(np.array(medians), np.array(negatives), np.array(sizes))
  
  variables = roughnessPenaltyIRLS(medians, negatives, sizes)
  
  if includeDecoys:
    evalScores = [x[0] for x in combined]
  else:
    evalScores = sorted(targetScores, reverse = True)
  
  if pi0 < 1.0:
    factor = pi0 * float(len(targetScores)) / len(decoyScores)
  else:
    factor = 1.0
  
  probs = factor * np.exp(list(map(lambda score : splineEval(score, medians, variables), evalScores)))
  probs = monotonize(probs)
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
  return list(map(lambda x : -1*np.log(x / (1 - x)), pvalues))
  
def monotonize(peps):
  newPeps = [0] * (len(peps)+1)
  if len(peps) > 0:
    for i, pep in enumerate(peps):
      newPeps[i+1] = min(1.0, max(newPeps[i], pep))
  return newPeps[1:]
  
def binData(combined, numBins = 500):
  binEdges = list(map(lambda x : int(np.floor(x)), np.linspace(0, len(combined), numBins+1)))
  bins = [combined[startIdx:endIdx] for startIdx, endIdx in zip(binEdges[:-1], binEdges[1:]) if startIdx < endIdx]
  prevMedian = None
  results = list()
  for b in bins:
    m = np.median([x[0] for x in b])
    numNegs = np.sum(1 for x in b if not x[1])
    numTot = len(b)
    if m != prevMedian:
      results.append([m, numNegs, numTot])
      prevMedian = m
    else:
      results[-1][1] += numNegs
      results[-1][2] += numTot
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

def iterativeReweightedLeastSquares(medians, negatives, sizes, variables, alpha, epsilon = 1e-8, tolerance = 0.001, maxiter = 50):
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
    if step < stepEpsilon:
      return (Q, R, g, w, z, gamma, p, gnew)
      
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

def splineEval(score, medians, variables):
  _, _, g, _, _, gamma, _, _ = variables
  #score = np.exp(score)
  n = len(medians)
  right = bisect.bisect_left(medians, score)
  
  if right == n:
    derl = (g[n - 1] - g[n - 2]) / (medians[n - 1] - medians[n - 2]) + (medians[n - 1] - medians[n - 2]) / 6 * gamma[n - 3]
    gx = g[n - 1] + (score - medians[n - 1]) * derl
    return gx
  
  if medians[right] == score:
    return g[right]
  
  if right > 0:
    left = right
    left -= 1
    dr = medians[right] - score
    dl = score - medians[left]
    gamr = gamma[right - 1] if right < (n - 1) else 0.0
    gaml = gamma[right - 1 - 1] if right > 1 else 0.0
    h = medians[right] - medians[left]
    gx = (dl * g[right] + dr * g[right - 1]) / h - dl * dr / 6 * ((1.0 + dl / h) * gamr + (1.0 + dr / h) * gaml)
    return gx
  else: # if right == 0
    derr = (g[1] - g[0]) / (medians[1] - medians[0]) - (medians[1] - medians[0]) / 6 * gamma[0]
    gx = g[0] - (medians[0] - score) * derr
    return gx

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
    
def unitTestScoreInput():
  import scipy.stats
  import matplotlib.pyplot as plt

  for s in range(3,4):
    np.random.seed(s)
    N = 5000
    #targetScores = np.round(np.random.normal(2.0,1,N) + np.random.normal(0.0,1,N), 1)
    #decoyScores = np.round(np.random.normal(0.0,1,N), 1)
    targetScores = np.random.normal(2.0,1,N) + np.random.normal(0.0,1,N)
    decoyScores = np.random.normal(0.0,1,N)
    _, peps1 = getQvaluesFromScores(targetScores, decoyScores, includePEPs = True, includeDecoys = True, tdcInput = True)
    _, peps2 = getQvaluesFromScoresQvality(targetScores, decoyScores, includePEPs = True, includeDecoys = True, tdcInput = True)
    
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
