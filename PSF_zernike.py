#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 16:10:31 2022

@author: shaohli
"""

''''PSF simulation with phase mask'''

import numpy as np
import tifffile as tf
from scipy import ndimage
from scipy import fftpack
import Zernike36 as Z
from PIL import Image as I
import matplotlib.pyplot as plt

from scipy.fftpack import fft2
from scipy.fftpack import fftshift, ifftshift

pi = np.pi
fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift



imgSize = 256
wavelength = 0.56 # microns
NA         = 0.8 # NA of the objective
nImmersion = 1 # Refractive index
pixelSize  = 0.1 # microns

kMax = 2 * pi / pixelSize     # Value of k at the maximum extent of the pupil function
kNA  = 2 * pi * NA / wavelength   #wavenumber


dk   = 2 * pi / (imgSize * pixelSize)      
kx = np.arange((-kMax + dk) / 2, (kMax + dk) / 2, dk)
ky = np.arange((-kMax + dk) / 2, (kMax + dk) / 2, dk)
KX, KY = np.meshgrid(kx, ky) # coordinate for the pupil function


maskRadius = kNA / dk # Radius of amplitude mask for defining the pupil
maskCenter = np.floor(0)
xv, yv       = np.meshgrid(np.arange(-imgSize/2, imgSize/2), np.arange(-imgSize/2, imgSize/2))
mask       = np.sqrt((xv - maskCenter)**2 + (yv- maskCenter)**2) < maskRadius

def getZArrWF(zarr):
    ph = np.zeros((imgSize, imgSize),dtype=np.float32)
    for j,m in enumerate(zarr):
        ph = ph + m*Z.Zm(j,rad=maskRadius,orig=None,Nx=imgSize)
    bpp = mask*np.exp(1j*ph)
    return bpp

def zslides_trans(img):
    nz, nx, ny = img.shape
    z_y = np.zeros((nz,ny))
    for i in range(nz):
        for j in range(ny):
            z_y[i,j] = max(img[i,:,j])
    return z_y

#the pupil
amp   = np.ones((imgSize, imgSize)) * mask
phase = 2j * np.pi * np.ones((imgSize, imgSize))
pupil = amp * np.exp(phase)


#Define aberration 
''' Zernike modes with single index now, with Noll ordering '''
'''zarr = [1, hor_tilt, ver_tilt, defocus, oblique astigmatism, 
            astigmatism, hor_coma, ver_coma, hor_trefoil, oblique trefoil, 
            oblique trefoil, spherical, oblique 2nd astigmatism, 2nd astigmatism, z4-4]'''
zarr = [0,0,0,0,1,
        0,0,0,0,0,
        0,0,0,0,0]  
bpp = getZArrWF (zarr)    
pupil = bpp*pupil
# tf.imshow(np.abs(bpp))
# tf.imshow(np.abs(pupil.real))

psf_c             = fftshift(fft2(ifftshift(pupil)))* dk**2
psf              = psf_c * np.conj(psf_c) # PSFA times its complex conjugate
# tf.imshow(np.real(psf), interpolation = 'nearest')

# Defocus from -3 micron to 3 micron
zrange = np.arange(-3, 3, 0.1)
PSF_stack      = np.zeros((zrange.size, imgSize, imgSize))

for ctr, z in enumerate(zrange):
    defocusPhaseAngle   = 1j * z * np.sqrt((2 * np.pi * nImmersion / wavelength)**2 - KX**2 - KY**2 + 0j)
    defocusKernel       = np.exp(defocusPhaseAngle)
    defocusPupil        = pupil * defocusKernel
    defocusPSFA         = fftshift(fft2(ifftshift(defocusPupil))) * dk**2   
    PSF_stack [ctr,:,:] = np.real(defocusPSFA * np.conj(defocusPSFA))


PSF_ZY = zslides_trans(PSF_stack)
# tf.imwrite('PSFstack.tif',PSF_stack)   # save as tiff

# Plot with zoom in
ZoomIn_l = int(np.floor(imgSize/2-40))
ZoomIn_r = int(np.floor(imgSize/2+40))


fig, axes = plt.subplots(1,5, figsize=(12, 8))
axes[0].imshow(PSF_stack[0,ZoomIn_l:ZoomIn_r,ZoomIn_l:ZoomIn_r],cmap='hot')
# axes[0].set_title("z = " + str(zrange[0]) + "um",fontsize = 10)
axes[0].axis('off')

axes[1].imshow(PSF_stack[int(len(zrange)/4),ZoomIn_l:ZoomIn_r,ZoomIn_l:ZoomIn_r],cmap='hot')
# axes[1].set_title("z ="  + str(zrange[int(len(zrange)/4)]) + "um",fontsize = 10)
axes[1].axis('off')

axes[2].imshow(PSF_stack[int(len(zrange)/2),ZoomIn_l:ZoomIn_r,ZoomIn_l:ZoomIn_r],cmap='hot')
axes[2].set_title("z = 0",fontsize =10)
axes[2].axis('off')

axes[3].imshow(PSF_stack[int((3*len(zrange))/4),ZoomIn_l:ZoomIn_r,ZoomIn_l:ZoomIn_r],cmap='hot')
# axes[3].set_title("z ="  + str(zrange[int(3*len(zrange)/4)]) ,fontsize = 10)
axes[3].axis('off')

axes[4].imshow(PSF_stack[int(len(zrange))-1,ZoomIn_l:ZoomIn_r,ZoomIn_l:ZoomIn_r],cmap='hot')
# axes[4].set_title("z ="  + str(zrange[int(len(zrange)-1)]), fontsize =10)
axes[4].axis('off')


ZoomIn_lz = int(np.floor(imgSize/2-len(zrange)/2))
ZoomIn_rz = int(np.floor(imgSize/2+len(zrange)/2))

fig, axes = plt.subplots(1,3, figsize=(12, 8))
axes[0].imshow(np.abs(pupil.real))
axes[0].set_title("pupil",fontsize = 15)
axes[0].axis('off')

axes[1].imshow(PSF_stack[int(len(zrange)/2),ZoomIn_l:ZoomIn_r,ZoomIn_l:ZoomIn_r],cmap='hot')
axes[1].set_title("X-Y",fontsize = 15)
# axes[1].axis('off')

axes[2].imshow(PSF_ZY[:,ZoomIn_lz:ZoomIn_rz],cmap='hot')
axes[2].set_title("X-Z",fontsize = 15)
# axes[2].axis('off')

plt.tight_layout()
plt.show()




