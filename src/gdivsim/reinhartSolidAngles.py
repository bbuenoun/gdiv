#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=========================================================================
Compute differential solid angles and their cosines per bin in a Reinhart 
angle basis as a function of subdivision factor MF. 
Derived from klemsSolidAngles.py, with loops collapsed into comprehensions.

$Id: reinhartSolidAngles.py,v 1.3 2019/11/14 01:01:22 u-no-hoo Exp u-no-hoo $

Roland Schregle (roland.schregle@gmail.com)
(c) Fraunhofer Institute for Solar Energy Systems
=========================================================================
"""


from numpy import (
   array,
   zeros,
   cos,
   sin,
   pi,
   arange,
   linspace,
   concatenate
)

from pprint import PrettyPrinter
from sys import argv



def reinhartSolidAngles(MF):
   """
   Compute differential solid angles and their corresponding cosines for each bin 
   in a Reinhart angle basis with subdivision factor MF (typically a power of 2).
   Returns solid angles and cosines in two separate arrays which may be
   multiplied by the caller to obtain projected solid angles.
   """
   nBins = MF**2 * 144 + 1    # Number of bins (= sum(azimuthDivs))
   
   # Polar angles at bin boundaries [rad]. These are distributed over the 
   # full hemisphere, then halved to obtain 1/2 division at the polar cap. 
   # Alpha = polar increment (dTheta) per division.   
   (polarAngles, alpha) = linspace(-0.5 * pi, 0.5 * pi, 2 * MF * 7 + 2, retstep = True)
   polarAngles = concatenate([zeros(1), polarAngles [polarAngles > 0]])
   nPolar = len(polarAngles)
   
   # Azimuth divisions [deg]
   azimuthDivs = [1] + [MF * i for i in [6, 12, 18, 24, 24, 30, 30] for r in range(MF)]

   # Polar angles of bin midpoints [rad]
   polarAngM = zeros(nPolar - 1)
   polarAngM [1:] = 0.5 * (polarAngles [1:-1] + polarAngles [2:])
   
   # ------------------------------------------------------------------------
   # Compute differential solid angles dOmega and their cosines
   dOmega = zeros(nBins)
   cosOmega = zeros(nBins)
   # Solid angle of spherical cap [c.f. Global Illumination Compendium (30)]
   dOmega [0] = 2 * pi * (1 - cos(polarAngles [1]))
   # Cosine term of integral over spherical cap
   cosOmega [0] = pi * (1 - cos(polarAngles [1])**2) / dOmega [0]
   # Differential solid angles [c.f. Global Illumination Compendium (22)]
   dOmega [1:] = [
      sin(polarAngM [t]) *       # Projection on azimuthal plane
      alpha *                    # dTheta
      2 * pi / azimuthDivs [t]   # dPhi
      for t in range(1, nPolar - 1) for p in range(azimuthDivs [t])
   ]
   # Cosines of the above (taken at midpoints)
   cosOmega [1:] = [
      cos(polarAngM [t]) 
      for t in range(1, nPolar - 1) for p in range(azimuthDivs [t])
   ]

   # ------------------------------------------------------------------------
   # Reinhart orders bins counterclockwise from the horizon to the zenith,
   # so reverse the arrays
   return (dOmega [::-1], cosOmega [::-1])
    
    
    
if __name__ == '__main__':
   if len(argv) > 1:
      (dOmega, cosOmega) = reinhartSolidAngles(int(argv [1]))
      pp = PrettyPrinter(indent = 3)
      print('\ndOmega = ')
      pp.pprint(dOmega)
      print('\ncosOmega = ')
      pp.pprint(cosOmega)
      # Sums converge to 2*PI resp. PI as MF -> inf, 
      # and discretisation error diminishes
      print('\nBins = %d\nSum = %f\nSum projected = %f' % 
            (len(dOmega), dOmega.sum(), (dOmega * cosOmega).sum()))
   else:
      print('%s <MF>' % argv [0])
      exit(-1)

