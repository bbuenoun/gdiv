#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=========================================================================
Estimate divergence in g-value for solar calorimeter from HDR camera
fisheye image of measurement setup. Outputs pixel indices with their 
associated radiance contributions and normalised coefficients.

$Id: gdivmeas_fullres.py,v 1.8 2020/02/03 23:47:34 u-no-hoo Exp u-no-hoo $

Roland Schregle (roland.schregle@gmail.com)
(c) Fraunhofer Institute for Solar Energy Systems
=========================================================================
"""



import os
import numpy as np
import traceback
from sys import argv
from numpy import pi, sin, cos, arccos, sum, any, sqrt, cross, dot, array, zeros, roll
from numpy.linalg import norm
from scipy.optimize import newton
#from numba import jit



DEBUG    = False
# Compensation for FP roundoff
EPSILON  = 1e-6

# Output paths for radiance contributions, normalised coefficients, and scalar irradiance
RADCONT  = 'radcontrib/%s.dat'
IRRAD    = 'irrad/%s.dat'
COEFF    = 'coeff/%s.dat'

# Command templates
# Dummy mode: dry run, print commands only
DUMMY    = False
# Get image resolution
GETINFO  = 'getinfo -d < %s'
# Output pixels with their contributions
PVALUE   = 'pvalue -o -h -H -b %s'
# Handy for checking pixel pos, dir and radiance in HDR image with <t>
XIMAGE   = 'ximage -opdv %s'



# ------------------- Pixel <--> index mapping stuff ----------------------
# Central angle alpha subtended by chord at height v.
# Note alpha covers the interval [0, 2*pi], and thus the entire fisheye
# image, where alpha = 0 coincides with the up vector.
alpha = lambda v, r: 2 * arccos(v / r) if v <= r else 0

# Area of circular segment at height v (= vertical index component)
segArea = lambda v, r: 0.5 * r**2 * (alpha(v, r) - sin(alpha(v, r)));

# Chord length at height v (= horizontal index component }
chordLen = lambda v, r: 2 * sqrt(r**2 - v**2) if v <= r else 0



#@jit
def vwRay (npix):
   """
   Map normalised pixel coordinates rpix=(u,v) in [0,1]^2 to corresponding 
   view ray (Dx, Dy, Dz) for an angular fisheye projection. Replaces former 
   external fisheyevwray.cal, since required internally by pixOmega().
   NOTE: This mapping assumes a LEFT-HANDED coordinate system, i.e. with the
   Z-axis (optical axis) pointing _into_ the image. Thus, directions outside
   the field of view (image corners) will have a negative Z-component.
   
   Returns view ray vector as numpy array.
   """
   (rx, ry) = npix
   # Convert to radial coords (relative to origin) on X, Y projection plane
   rx = 2 * rx - 1
   ry = 2 * ry - 1
   # Equiangular projection; angle theta to optical (Z) axis is proportional 
   # to radial distance dr
   dr = sqrt(rx**2 + ry**2)
   Dz = cos(.5 * pi * dr)
   # Project Dz onto Dx and Dy
   pz = sqrt(1 - Dz**2) / dr if dr > EPSILON else 0
   Dx = rx * pz
   Dy = ry * pz
   return array([Dx, Dy, Dz])



#@jit
def pixOmega(pix, radius):
   """
   Calculate solid angle of pixel coordinates pix=(x,y) in fisheye image 
   based on spherical excess for an n-sided polyhedron:
      omega = sum(alpha_i) - (n - 2) * PI,
   where alpha_i is the angle between adjacent planes passing between the
   origin and the i-th polyhedron edge.
   """
   # Normalised pixel quadrilateral vertices (assume x,y at centre)
   (x, y) = pix
   res = 2 * radius
   (x1, y1, x2, y2) = (
      (x - 0.5) / res, (y - 0.5) / res, 
      (x + 0.5) / res, (y + 0.5) / res
   )   
   pixVerts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
   # Edges of 4-sided frustum subtended by pixel vertices, arranged as matrix
   pixEdges = array([vwRay(v) for v in pixVerts])
   if DEBUG and any(pixEdges [:, 2] < EPSILON):
      # TODO: How to handle vertices outside FoV?   
      print('WARNING: Pixel quadrilateral partially outside FoV at (%d, %d)' % pix)
   # Get 2nd edges forming planes by subtracting adjacent pixel vertex rays
   # (note subtraction of pixEdges from itself after shift/wrap by 1 row)
   pixEdges2 = roll(pixEdges, shift = -1, axis = 0) - pixEdges
   # Get frustum plane normals
   pixNormals = cross(pixEdges, pixEdges2)
   # pixEdges aren't normalised, so normalise now; bail out if zero length
   pixNorms = norm(pixNormals, axis = 1, keepdims = True)
   if any(pixNorms < EPSILON):
      print('WARNING: Degenerate pixel at %r' % pix)
      return 0
   else:
      pixNormals /= pixNorms
      
   # Sum angles between adjacent plane normals to get spherical excess
   omega = (
      sum(arccos(sum(pixNormals * roll(pixNormals, -1, axis = 0), axis = 1)))
      - 2 * pi
   )

   return abs(omega)
      
      

def pix2Idx (pix, radius):
   """
   Map pixel coordinates pix=(x,y) in fisheye image of specified radius 
   (in pixels) to their serialised index. Indexing starts from the top 
   of the image and extends left to right towards the bottom.
   The index corresponds to the area of the circular segment 
   (see http://mathworld.wolfram.com/CircularSegment.html) down to vertical 
   offset v, plus horizontal offset u.
   
   NOTE: Coordinates outside the radius are mapped to index -1.
   """
   # Get radial coords (u,v), check if inside radius
   (u, v) = pix
   u -= radius
   v -= radius
   if u**2 + v**2 + EPSILON > radius**2:
      return -1
      
   # Add vertical index component for previous line (= segment area) to 
   # horizontal offset into current line (= partial chord length)
   return (
      int(segArea(v + 1, radius) + 0.5) + 
      int(0.5 * chordLen(v + 1, radius) + u + 1.5)
   )
   


def idx2Pix (idx, radius):
   """
   Map pixel index in fisheye image of specified radius in pixels (= 1/2 * 
   resolution per axis) to its coordinates (x,y). The index 
   represents the circular segment area containing the encoded pixel in 
   its chord (see http://mathworld.wolfram.com/CircularSegment.html).
   
   There is no closed form solution for obtaining the central angle
   subtended by the circular segment from the index and the radius alone, 
   so Newton-Raphson optimization is applied to numerically solve for it. 
   """
   # Check for idx in range
   if idx < 0 or idx > pi * radius**2:
      return None
      
   # Function to solve for alpha
   f  = lambda alpha, idx, radius: 2 * idx / radius**2 + sin(alpha) - alpha
   # Derivative of above
   df = lambda alpha, idx, radius: cos(alpha) - 1
   # Initial guess for alpha
   alpha0 = pi
   # Find alpha for circular segment area = idx
   alphaOpt = newton(f, x0 = alpha0, args = (idx - 1, radius), fprime = df)   
   
   # Pixel's vertical radial component v
   v = int(radius * cos(alphaOpt / 2))
   # Hack in offset for upper hemicircle (not clear yet *why* we need this)
   v += 1 if alphaOpt < pi else 0   
   #print(' %f %f ' % (alphaOpt, alpha(v, radius)), end = '')
   # Subtract vertical index offset and start of current line (chord)
   # from index to get horizontal radial component u
   u = (
      idx - 1 - 
      int(segArea(v, radius) + 0.5) - 
      int(0.5 * chordLen(v, radius) + 0.5)
   )
   
   # Translate to image coords
   return (u + radius, v + radius - 1)
      
      

def doItBabee (commands, echo = True, dummy = False, pipe = False):
   """
   Execute each command in a subshell or all at once in a pipe.
   echo:    toggle output of each command to stdout.
   dummy:   dry run, echo only and DON'T actually do it, babee. 
   pipe:    run all commands at once as part of a pipe.
   
   If dummy==False: pipe output as file (pipe==True), or exit status 
   of last command (pipe==False).
   If dummy==True: output from /dev/null (pipe==True), or 0 (pipe==False).
   """
   NULL = '/dev/null'
   cList = [(' |\n').join(commands)] if pipe else commands
   
   for c in cList:
      if echo or dummy:
         print(c)
      if not dummy:
         ret = os.popen(c) if pipe else os.system(c)    
      else:
         ret = open(NULL) if pipe else 0
   return ret
   
     

def getCoeffs (hdrFile):
   """
   Extract measured radiance contributions and normalised coefficients 
   from camera image. 
   """
   # Camera image
   # Instantiate filename for current image
   basename = os.path.splitext(os.path.basename(hdr)) [0]
   radContFile  = RADCONT % basename
   irradFile = IRRAD % basename
   coeffFile = COEFF % basename
   # Instantiate commands for current image
   getinfo  = GETINFO % hdrFile
   pvalue   = PVALUE % hdrFile
   # Get fisheye image radius in pixels from resolution 
   # (= MINIMUM of either axis if distorted)
   try:
      resStr= doItBabee([getinfo], dummy = DUMMY, pipe = True).read().split()
      rad = int(0.5 * min(int(resStr [1]), int(resStr [3]))) if resStr else 0
   except Exception as ex:
      print("ERROR extracting resolution from %s: %s" % (hdr, str(ex)))
      exit(-1)

   # Launch commands, get output at end of pipe (empty in dummy mode)
   pipeOut  = doItBabee([pvalue], dummy = DUMMY, pipe = True)
   
   # Preallocate bin arrays based on radius; mark empty with -1
   numIdx = int(pi * rad**2 + 0.5)
   radContribs = array([-1] * numIdx, dtype = np.float)
   omegas = array([-1] * numIdx, dtype = np.float)

   # Read pixel coords and radiance contributions per row, accumulate in bins
   for line in pipeOut:
      rec = line.split()
      pix = (int(rec [0]), int(rec [1]))      
      contrib = float(rec [2])
      # Map pix to its index = bin#
      idx = pix2Idx(pix, rad)
      # Skip invalid indices < 0
      if idx >= 0:
         # Populate bin #idx with contrib
         radContribs [idx] = contrib                  
         # Map pixel to view ray (assuming it passes thru pixel centre), 
         # get cosine from Z-component
         (_, _, Dz) = vwRay((0.5 * pix [0] / rad, 0.5 * pix [1] / rad))
         if Dz < 0:
            raise Exception('Valid index %d outside FoV (Dz = %f)' % (idx, Dz))
         # Populate bin #idx with projected solid angle
         omegas [idx] = pixOmega(pix, rad) * Dz
         if DEBUG:
            # Diagnostics for pixel index and its reverse
            print(
               "%r\t-> %d\t-> %r" % 
               (pix, idx, idx2Pix(idx, rad))
            )         

   pipeOut.close()
   if DEBUG:
      print("Sum proj. omega = %f" % (sum(omegas)))
      
   # Drop empty bins from arrays
   nonEmpty = radContribs >= 0
   radContribs = radContribs [nonEmpty]
   omegas = omegas [nonEmpty]
   if not len(radContribs):
      # Bail out if no output (-> dummy mode)
      return
      
   # Dump radiance contributions to file
   np.savetxt(radContFile, radContribs, fmt = '%g')
   # Weight radiance contribs by projected solid angle = irradiance contribs, 
   # dump to file
   irradContribs = radContribs * omegas
   irrad = irradContribs.sum()
   with open(irradFile, 'w') as f:
      f.write('%s\n' % irrad)
   
   # Normalise irradiance contribs = coefficients, dump to file
   coeffs = irradContribs / irrad
   np.savetxt(coeffFile, coeffs, fmt = '%g')
   
         
         
if __name__ == "__main__":
   if len(argv) < 2:
      print("Usage: %s <img1.hdr> .. <imgN.hdr>" % argv [0])
      exit(-1)
      
   for hdr in argv [1:]:
      try:
         getCoeffs(hdr)
      except Exception as ex:
         print("ERROR extracting contributions/coefficients from %s: %s" % (hdr, str(ex)))
         traceback.print_exc()
         

