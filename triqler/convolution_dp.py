from __future__ import print_function

import sys
import numpy as np

def main(argv):
  np.random.seed(2)
  numPoints = 1001
  
  xs = np.linspace(-5, 5, numPoints)
  probs = generateData(numPoints)
  convProbs = convolveProbs(probs)
  
  print(np.sum(convProbs))
  plt.imshow(convProbs, interpolation = 'none')
  plt.show()  
  
def generateData(numPoints):
  probs = list()
  for i in range(4):
    p1 = np.random.rand(numPoints)
    p1 /= np.sum(p1)
    probs.append(p1)
  return probs

def convolveProbs(probs):
  numPoints = len(probs[0])
  convProbs = np.diag(probs[0])

  for p1 in probs[1:]:
    convProbsCopy = np.copy(convProbs)
    convProbs = np.zeros((numPoints, numPoints))
    rowCumSums = np.zeros((numPoints, numPoints))
    for j in range(numPoints):
      rowCumSums[:j, j] = np.cumsum(convProbsCopy[1:j+1, j][::-1])[::-1]
    for i in range(numPoints):
      convProbs[i, i:] += convProbsCopy[i, i:]*np.cumsum(p1[i:])
      convProbs[i, i:] += rowCumSums[i, i:]*p1[i]
      convProbs[i, i+1:] += np.cumsum(convProbsCopy[i, i:-1])*p1[i+1:]
  return convProbs

if __name__ == "__main__":
  main(sys.argv[1:])
