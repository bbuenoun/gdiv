#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=========================================================================
Cluster coefficients according to g-value symmetry defined in module 
gdivsymm.py, accumulate and weight them for each cluster, then normalise 
over all clusters.

$Id: gdivclust.py,v 1.8 2020/02/09 20:32:12 u-no-hoo Exp u-no-hoo $

Roland Schregle (roland.schregle@gmail.com)
(c) Fraunhofer Institute for Solar Energy Systems
=========================================================================
"""


import os
import numpy as np
from pprint import PrettyPrinter
from sys import argv, stdout, stderr


# Class instance encapsulating symmetry criteria and coefficient clustering
from gdivsymm import (
   KlemsSymmetry, 
   ReinhartRotSymmetry, ReinhartCurveSymmetry, ReinhartProfSymmetry,
   FullResSymmetry
)
#symm = ReinhartRotSymmetry()
symm = FullResSymmetry()


DEBUG = True


if __name__ == "__main__":
   if len(argv) < 3:
      print(
         "%s coeffFile_0 [coeffFile_1 ... coeffFile_n-1] outFile" % argv [0],
         file = stderr
      )
      exit(-1)

   # Check we don't clobber an existing output file; this prevents accidentally
   # clobbering an input file if the user forgets to specify the output file!
   # Do this _before_ any lengthy clustering calcs.
   if os.path.exists(argv [-1]):
      print('Output file %s exists, aborting' % argv [-1]);
      exit(-1)
   
   # Number of coefficient files on cmdline
   numCoeffFiles = len(argv) - 2
   numClust = len(symm.clusterCoeffs(argv [1]))
   clustArr = np.zeros((numCoeffFiles, numClust))
   pp = PrettyPrinter()            
   for (i, coeffFile) in enumerate(argv [1:-1]):
      try:
         clustVec = symm.clusterCoeffs(coeffFile)
         clustVecShape = (numCoeffFiles, len(clustVec))            
         if not clustArr.shape == clustVecShape:
            # Cluster dimensions is not constant
            raise ValueError(
               "Cluster dimensions don't match coefficient files; expected %s, got %s" % 
               (str(clustArr.shape), str(clustVecShape))
            )
         else:
            # Accumulate clustered coeffs in corresponding row
            clustArr [i, :] = clustVec
            
         if DEBUG:
            print('%s:' % coeffFile)
            pp.pprint(clustVec)
            print()
      except Exception as ex:
         print("Error clustering coefficients: %s" % str(ex), file = stderr)
         exit(-1)
         
   np.array2string(clustArr)
   # Dump clustered coefficient array to stdout
   np.savetxt(argv [-1], clustArr)
         
#~ if __name__ == "__main__":
   #~ # Number of coefficient files on cmdline
   #~ numCoeffFiles = len(argv) - 1
   #~ clustArr = np.zeros((numCoeffFiles, numCoeffFiles))   
      #~ 
   #~ if not numCoeffFiles:
      #~ print("%s coeffFile_0 [coeffFile_1 ... coeffFile_n-1]" % argv [0])
      #~ exit(-1)
   #~ 
   #~ for (i, coeffFile) in enumerate(argv [1:]):
      #~ try:
         #~ clustVec = clusterCoeffs(coeffFile)
         #~ clustVecShape = (numCoeffFiles, len(clustVec))
         #~ if not clustArr.shape == clustVecShape:
            #~ # Cluster dimensions differ from number of incident angles implied by coeff file sequence
            #~ raise ValueError(
               #~ "Cluster dimensions don't match coefficient files; expected %s, got %s" % 
               #~ (str(clustArr.shape), str(clustVecShape))
            #~ )
         #~ else:
            #~ # Accumulate clustered coeffs in corresponding row
            #~ clustArr [i, :] = clustVec
      #~ except Exception as ex:
         #~ print("Error clustering coefficients: %s" % str(ex))
         #~ exit(-1)
   #~ np.array2string(clustArr)
   #~ # Dump clustered coefficient array to stdout
   #~ np.savetxt(stdout, clustArr)

