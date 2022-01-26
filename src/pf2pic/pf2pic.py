#!/usr/bin/env python

"""
=========================================================================
Batch convert and crop PF images from luminance camera to RADIANCE HDR 
in units of radiance, using custom luminous efficacy.
Based on Shashanka's original pf2pic.py.

$Id: pf2pic.py,v 1.1 2020/01/12 00:06:26 u-no-hoo Exp u-no-hoo $

Roland Schregle (roland.schregle@gmail.com)
(c) Fraunhofer Institute for Solar Energy Systems
=========================================================================
"""

import os
import sys


# Files/paths
HDRDIR   = 'hdr'
FALSEDIR = 'false'

# Luminous efficacy in lm/W
LUMEFFIC = 91.40
VIEW     = '-vta -vh 180 -vv 180 -vp 0 0 0 -vd 0 0 1 -vu 0 1 0'
PFTOPIC  = 'pftopic -k %d %s < %s | '

# Find crop box by skipping runs of zeros and extracing min/max in X/Y;
# neaten removes redundant space for isolating coords via cut.
# Obviously this would be far more efficient in Python...
PVALUE   = 'pvalue -b -H -h -u | neaten 0' 
MINX     = PVALUE + ' | cut -d\  -f1 | total -l'
MAXX     = PVALUE + ' | cut -d\  -f1 | total -u'
MINY     = PVALUE + ' | cut -d\  -f2 | total -l'
MAXY     = PVALUE + ' | cut -d\  -f2 | total -u'

# Crop with pcompos
PCOMPOS  = 'pcompos -x %d -y %d - -%d -%d > %s'

# Falsecolour from HDR
FALSECOL = 'falsecolor -ip %s -l W/sr/m^2 -m 1 -lw 150 -s 1e5 -n 5 -log 5 | ra_tiff -z - %s'


if len(sys.argv) > 1:
   for pf in sys.argv [1:]:
      try:
         pfToPic = PFTOPIC % (LUMEFFIC, VIEW, pf)
         # Get crop box
         crop = []
         for bound in (MINX, MINY, MAXX, MAXY):
            cmd = pfToPic + bound
            #print(cmd)
            crop.append(int(os.popen(cmd).read()))
      
         minX = crop [0]
         minY = crop [1]
         maxX = crop [2]
         maxY = crop [3]
         sizeX = maxX - minX + 1
         sizeY = maxY - minY + 1
         print('Cropping to %dx%d+%d+%d' % (sizeX, sizeY, minX, minY))
         
         # Convert to HDR and crop
         hdrFile = os.path.splitext(os.path.basename(pf)) [0] + '.hdr'
         hdrPath = os.path.join(HDRDIR, hdrFile)
         cmd = pfToPic + PCOMPOS % (sizeX, sizeY, minX, minY, hdrPath)
         print(cmd)
         os.system(cmd)
         
         # Render falsecolour
         falseFile = os.path.splitext(os.path.basename(pf)) [0] + '.tif'
         falsePath = os.path.join(FALSEDIR, falseFile)
         cmd = FALSECOL % (hdrPath, falsePath)
         print(cmd)
         os.system(cmd)
      
      except Exception as ex:
         print('Failed command: %s' % cmd)
         print(str(ex))
         exit(-1)

else:
   print(sys.argv [0] + ' <pfFile1> ... <pfFileN>')
   exit(-1)
