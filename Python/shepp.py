# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# This file is part of the TIGRE Toolbox
#
# Copyright (c) 2015, University of Bath and
#                     CERN-European Organization for Nuclear Research
#                     All rights reserved.
#
# License:            Open Source under BSD.
#                     See the full license at
#                     https://github.com/CERN/TIGRE/blob/master/LICENSE
#
# Contact:            tigre.toolbox@gmail.com
# Codes:              https://github.com/CERN/TIGRE/
# Coded by:           Alexey Smirnov
# --------------------------------------------------------------------------

#%%Initialize
import tigre
import numpy as np
import skimage
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale
from tigre.utilities import sample_loader
from tigre.utilities import CTnoise
import tigre.algorithms as algs
from tigre.utilities.Measure_Quality import Measure_Quality
import tigre.utilities.gpu as gpu
import matplotlib.pyplot as plt
from tigre.utilities import sl3d
from tigre.utilities.im3Dnorm import im3DNORM

#%% Geometry
# geo = tigre.geometry(mode="cone", default=True, nVoxel=np.array([256, 256, 256])
# geo = tigre.geometry_default(high_resolution=False)

## FAN BEAM 2D

geo = tigre.geometry(default=True)
# VARIABLE                                   DESCRIPTION                    UNITS
# -------------------------------------------------------------------------------------
# Image parameters
geo.nVoxel = np.array([1, 256, 256])  # number of voxels              (vx)
geo.sVoxel = np.array([1, 256, 256])  # total size of the image       (mm)
geo.dVoxel = geo.sVoxel / geo.nVoxel  # size of each voxel            (mm)

geo.mode = "cone"

#%% Load data and generate projections
# define angles
angles = np.linspace(0, 2/3 * np.pi, 120)

# Load phantom data
phantom_type = ("yu-ye-wang")

# shepp3d = sl3d.shepp_logan_3d([256, 256, 256], phantom_type=phantom_type)
# shepp = shepp3d[128,:,:].reshape(1, 256, 256) 
shepp = np.float32(shepp_logan_phantom())
shepp = rescale(shepp, scale=256/400, mode='reflect')
shepp = shepp.reshape(1, 256, 256) 

print(shepp.shape)

# generate projections
projections = tigre.Ax(shepp, geo, angles)
# add noise
# noise_projections = CTnoise.add(projections, Poisson=1e5, Gaussian=np.array([0, 10]))
noise_projections = projections

#%% Plot
plt.imsave('shepp-logan.png', shepp[0])

# tigre.plotProj(proj)
# tigre.plotImg(fdkout)
