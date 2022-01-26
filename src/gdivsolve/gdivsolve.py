#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=========================================================================
Estimate divergence in g-value from solar calorimeter based on Klems 
luminance coefficients. These are accumulated into coefficients b_i,j for
rotationally symmetric incident angles theta_i for each measured/simulated 
lamp angle theta_j to represent the measured/simulated g-values g'(theta_i) 
as a linear combination of the sought g-values g(theta_i) from a parallel 
source:

   g'(theta_i) ≈ b_i,0 g(theta_0) + ... + b_i,j-1 g(theta_j-1),
   
or in matrix form: 

   G' ≈ B G,
   
This system of linear equations is then solved for vector G in a separate 
script to obtain an estimate for the parallel g-values at the measured
incident directions theta_i.

For theory and derivation of this approach, refer to:

   Kuhn, Tilmann E. "Solar Control: A general Evaluation Method for 
   Façades with Venetian Blinds or other Solar Control Systems". 
   Energy and Buildings 38(6) (2005) 648-660.
   DOI: 10.1016/j.enbuild.2005.10.002.

$Id: gdivsolve.py,v 1.3 2019/11/19 12:51:52 u-no-hoo Exp u-no-hoo $

Roland Schregle (roland.schregle@gmail.com)
(c) Fraunhofer Institute for Solar Energy Systems
=========================================================================
"""



import os
import numpy as np
from numpy import linalg
from scipy import optimize
from sys import argv, stdout, stderr


         
if __name__ == "__main__":
   # Get coefficient matrix and RHS vector (divergent g-values)
   argc = len(argv) - 1
   if argc < 2:
      print("%s <coeffFile> <gvalFile>" % argv [0], file = stderr)
      exit(-1)
      
   coeffFile = argv [1]
   gvalFile  = argv [2]
   
   try:
      coeffs = np.loadtxt(coeffFile)
   except Exception as ex:
      print(
         "Error loading coefficients from %s: %s" % (coeffFile, str(ex)), 
         file = stderr
      )
      exit(-1)
   
   try:
      gdivVals = np.loadtxt(gvalFile)
   except Exception as ex:
      print(
         "Error loading divergent g-values from %s: %s" % (gvalFile, str(ex)),
         file = stderr
      )
      exit(-1)
      
   try:
      # Solve linear equation for parallel g-values
      gparValsLin = linalg.solve(coeffs, gdivVals)
      #gparValsLsq = linalg.lstsq(coeffs, gdivVals, rcond = None)
      
      # Objective function to minimise
      objFunc = lambda x: linalg.norm(coeffs.dot(x) - gdivVals)
      # Initial guess for parallel g-values = divergent g-values
      gparVals0 = gdivVals
      # Boundary values; clamp in range [0..1] for plausibility
      gvalBounds = [(0, 1) for g in gdivVals]
      gparValsMin = optimize.minimize(
         objFunc, gparVals0, method = 'L-BFGS-B', bounds = gvalBounds
      )
      if not gparValsMin.success:
         # Failed minimisation, raise exception
         raise Exception(gparValsMin.message)
         
   except Exception as ex:
      # Singular or non-square matrix
      print("Error solving for parallel g-values: %s" % str(ex), file = stderr)
      exit(-1)

   print("\nLinear:\t\t%s" % str(gparValsLin))
   #print("\nLeast squares:\t%s" % str(gparValsLsq [0]))
   #print("Residual:\t%s" % str(gparValsLsq [3]))
   print("\nMinimisation:\t%s" % str(gparValsMin.x))
   print("Objective func: %f" % gparValsMin.fun)
   print("Iterations: %d" % gparValsMin.nit)

   #np.savetxt(stdout, gparVals)

