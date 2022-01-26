#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=========================================================================
Simulate divergence in g-value from solar calorimeter with RADIANCE model
of measurement setup. Derived from Shashanka Sri Nagaveera Sunkara's 
original RADIANCE model and script image.py.

Outputs radiance contributions and normalised coefficients on the observer
plane discretised in a Reinhart angle basis, corresponding to a sensor's 
fisheye view. 

$Id: gdivsim-reinhart.py,v 1.5 2019/11/29 14:56:20 u-no-hoo Exp u-no-hoo $

Roland Schregle (roland.schregle@gmail.com)
(c) Fraunhofer Institute for Solar Energy Systems
=========================================================================
"""



import os
import numpy as np
from sys import stderr
from reinhartSolidAngles import reinhartSolidAngles



RAD      = 'setup-%s%d.rad'
# Files (views, lamp configurations, incident angles)
VIEWDIR  = 'vf'
VIEWS    = ['SP5.vf']
#VIEWS    =  ['SP1.vf', 'SP3.vf', 'SP5.vf', 'SP7.vf', 'SP10.vf', 'SP11.vf']
LAMPCONF = ['3h']
#LAMPCONF = ['4q']
#THETAIN = [0]
#~ THETAIN  = [86]
# Reinhart subdivision factor (power of 2)
MF = 4
alpha = 90./(MF*7 + .5)
r_row = int(90/alpha)
THETAIN  = []
THETAIN.append(0)
for i in range(1,r_row+1):
    THETAIN.append(int(i*alpha))
OCT      = 'setup-%s%d.oct'
HDR      = 'hdr/%s.hdr'
FALSE    = 'false/%s.tif'
# Output directories for radiance contributions, normalised coefficients, and scalar irradiaance
RADCONT  = 'radcontrib/%s.dat'
IRRAD    = 'irrad/%s.dat'
COEFF    = 'coeff/%s.dat'
SENSFILE = 'sensor.dat'

# Rendering params
# Dummy mode: dry run, print commands only
DUMMY    = False
# Generate HDR rendering in addition to binning coefficients
RENDER   = False
RES      = (400, 400)
NUMSAMP  = 100000
NPROC    = 4
MOD      = 'sensormat'
# "Quality" setting
HQ       = True
if HQ:
   # High quality (slow!)
   # Note ambient cache options -aa and -ar are ignore by rcontrib
   RTOPTS   = ('-n %d -ds 0.02 -dc 1 -dt 0 -dj 0.5 -st 0 -ss 64 '
               '-ab 5 -aa 0.01 -ar 1024 -ad 4096 -as 1024 -lw 1e-4' % NPROC)
else:
   # Quick & dirty
   RTOPTS   = ('-n %d -ab 1' % NPROC)

# Command templates
NICE     = 'nice '
# Generate rays on sensor plane
RSENSOR  = 'rsensor -h -rd %d %s %s %s %s .'
# Reinhart binning (MF = subdiv factor, rN = normal, U = up)
BINNING  = "-f reinhartb-gdiv.cal -e 'MF:%d;rNx=%s;rNy=%s;rNz=%s;Ux=%s;Uy=%s;Uz=%s' -bn Nrbins -b rbin"
# Sum binned contributions
RCONTRIB = NICE + 'rcontrib -faa -h %s -c %d -V+ %s -m %s %s'
# Generate rays for current sensor pos
VWRAYS   = 'vwrays -ff -c %d -vf %s -pj 0.5 -x %d -y %d'
# Render HDR with rtrace (1 ray / pixel)
RTRACE   = NICE + 'rtrace -ffc `%s -d` %s %s > %s'
# Render HQ HDR with rcontrib (NUMSAMP rays/pixel, slow!)
RCONTHDR = NICE + 'rcontrib -fac `%s -d` %s -c %d -V+ -fo -o %s -m %s %s'

# RADIANCE definition of a dummy sensor 'bubble' (lens?) surrounding the
# viewpoint to exclusively register ray intersections through this modifier.
# Its position is instantiated for each viewpoint.
# This is modified by a brightfunc that crops rays incident outside the
# 180° field of view, to match the corresponding fisheye camera image.
# (Alternatively, the view rays can be filtered beforehand with VWRAYSCROP)
SENSOR   = ('void brightfunc cropfunc 2 cropS crop180.cal 0 0 '
            'cropfunc trans sensormat 0 0 7 1 1 1 0 0 1 1 '
            'sensormat sphere sensor 0 0 4 %s %s %s 1e-4')
OCONV    = 'oconv - %s > %s'
FALSECOL = 'falsecolor -ip %s -pal hot -s 2000000 -log 4 | ra_tiff -z - %s'

# Reshape coeffs in rows, keep only one channel (=monochrome)
RCOLLATE = 'rcollate -h -oc 1 | cut -d\  -f1'

# FOR REFERENCE ONLY (not used)
# Postcrop regions in HDR image beyond 180° FoV
PCROP180 = "pcomb -e 'lo=if(Dy(1),0,li(1))' %s > %s"
# Filter VWRAYS output for rays beyond 180°; these have their direction vectors set to 0.
# rcontrib ignores these and just outputs zero. Note this requires ASCII I/O (vwrays -fa)!
# This is an alternative to the cropfunc in SENSOR, but isn't conveniently built into the octree.
VWRAYSCROP = "rcalc -e '$1=$1;$2=$2;$3=$3;$4=if(-$5,$4,0);$5=if(-$5,$5,0);$6=if(-$5,$6,0)'"



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
         print(c, file = stderr)
      if not dummy:
         ret = os.popen(c) if pipe else os.system(c)    
      else:
         ret = open(NULL) if pipe else 0
   return ret
   
     

def getCoeffs (view, lampConf, thetaIn, render = False):
   """   
   Extract simulated radiance contributions and normalised coefficients 
   for current viewpoint, lamp configuration and incident 
   angle theta. Optionally generate a rendered HDR image.   
   """
   # Instantiate filenames for current case
   viewFile = os.path.join(VIEWDIR, view)
   basename = lampConf + str(thetaIn) + os.path.splitext(view) [0]
   radContFile = RADCONT % basename
   irradFile = IRRAD % basename
   coeffFile = COEFF % basename
   radFile  = RAD % (lampConf, thetaIn)
   octFile  = OCT % (lampConf, thetaIn)
   
   # Parse view point, direction and up
   try:
      viewArgs  = open(viewFile).read().split()
   except Exception as ex:
      print("Error reading view file %s: %s" % (viewFile, str(ex)), file = stderr)
      return None

   try:
      vPnt = viewArgs.index('-vp')
      vDir = viewArgs.index('-vd')
      vUp  = viewArgs.index('-vu')
   except Exception as ex:
      print(
         'Missing view point/direction/up in view file %s: %s:' % (viewFile, str(ex)),
         file = stderr
      )
      return None

   # Instantiate commands for current case
   oconv = OCONV % (radFile, octFile)
   vPnt = viewArgs [vPnt: vPnt + 4]
   vDir = viewArgs [vDir: vDir + 4]
   vUp  = viewArgs [vUp: vUp + 4]
   rsensor = RSENSOR % (NUMSAMP, ' '.join(vPnt), ' '.join(vDir), ' '.join(vUp), SENSFILE)
   binning  = BINNING % tuple([MF] + vDir [1:] + vUp [1:])
   rcontrib = RCONTRIB % (RTOPTS, NUMSAMP, binning, MOD, octFile)

   # Generate octree with sensor bubble surrounding viewpoint
   sensor   = 'echo ' + SENSOR % tuple(vPnt [1:])
   doItBabee([sensor, oconv], dummy = DUMMY, pipe = True).close()
   # Feed sample rays into rcontrib, sum bins and reshape output contributions into pipe
   rcPipe = doItBabee([rsensor, rcontrib, RCOLLATE], dummy = DUMMY, pipe = True)
   if DUMMY:
      return   
      
   # Read radiance contributions from pipe into array
   radContrib = np.loadtxt(rcPipe)
   rcPipe.close()      
   if not radContrib.any():
      raise Exception('zero contributions')

   # Get differential solid angles and cosines of bins
   (dOmega, cosOmega) = reinhartSolidAngles(MF)
   # Normalise radiance contributions according to differential solid angles
   # to compensate for sampling bias, dump to file
   radContrib /= dOmega / dOmega.sum()
   np.savetxt(radContFile, radContrib, fmt = '%g')
   
   # Weight radiance contributions by projected solid angle to obtain
   # irradiance contributions, dump to file
   irradContrib = radContrib * dOmega * cosOmega
   irrad = irradContrib.sum()
   with open(irradFile, 'w') as f:
      f.write('%s\n' % irrad)
   
   # Normalise irradiance contributions to obtain coefficients, dump to file
   coeffs = irradContrib / irrad
   np.savetxt(coeffFile, coeffs, fmt = '%g')
      
   # Optionally generate HDR rendering
   if render:
      # Instantiate files/cmds for current view
      hdrFile  = HDR % basename
      falseFile = FALSE % basename   
      falsecol = FALSECOL % (hdrFile, falseFile)
      if True:
         vwrays   = VWRAYS % ((NUMSAMP, viewFile,) + RES)
         render   = RCONTHDR % (vwrays, RTOPTS, NUMSAMP, hdrFile, MOD, octFile)
      else:
         vwrays   = VWRAYS % ((1, viewFile,) + RES)
         render   = RTRACE % (vwrays, RTOPTS, octFile, hdrFile)
      doItBabee([vwrays, render], dummy = DUMMY, pipe = True).close()


         
if __name__ == "__main__":
   for lampConf in LAMPCONF:
      for thetaIn in THETAIN:
         for view in VIEWS:
            try:
               getCoeffs(view, lampConf, thetaIn, RENDER)
            except Exception as ex:
               print(
                  'ERROR extracting contributions/coefficients for lamp config %s, theta %s: %s' % 
                  (lampConf, thetaIn, str(ex)),
                  file = stderr
               )


