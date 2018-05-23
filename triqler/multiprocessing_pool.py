from __future__ import print_function

import sys
from multiprocessing import Pool

class MyPool:
  def __init__(self, processes=1):
    self.pool = Pool(processes)
    self.results = []
  
  def applyAsync(self, f, args):
    r = self.pool.apply_async(f, args)
    self.results.append(r)
    
  def checkPool(self, printProgressEvery = -1):
    try:
      outputs = list()
      for res in self.results:
        outputs.append(res.get(0xFFFFFFFF))
        if printProgressEvery > 0 and len(outputs) % printProgressEvery == 0:
          print(len(outputs),"/", len(self.results), "%.2f" % (float(len(outputs)) / len(self.results) * 100) + "%")
      self.pool.close()
      self.pool.join()
      return outputs
    except KeyboardInterrupt:
      print("Caught KeyboardInterrupt, terminating workers")
      self.pool.terminate()
      self.pool.join()
      sys.exit()

def addOne(i):
  return i+1

def unitTest():
  pool = MyPool(4)  
  for i in range(20):
    pool.applyAsync(addOne, [i])
  results = pool.checkPool()
  print(results)
