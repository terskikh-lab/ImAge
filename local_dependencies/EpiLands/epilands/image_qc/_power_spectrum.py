import logging
import numpy as np
from scipy.stats import binned_statistic
from typing import Tuple
import os

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)

## Image Quality Control


def power_spectrum(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/
    """
    image = image.copy()
    if image.shape[0] != image.shape[1]:
        logger.warning(f"Image is not square [{image.shape[0]},{image.shape[1]}]")
        minimum_edge_size = min(image.shape)
        logger.warning(
            f"Cropping image to dimensions [{minimum_edge_size},{minimum_edge_size}]"
        )
        image = image[:minimum_edge_size, :minimum_edge_size]
    # get the edge len in px
    npix = image.shape[0]
    ## ~~ Taking the Fourier transform ~~ ##
    # Calculate the fourier transform of the image
    fourier_image = np.fft.fftn(image)
    # Calculate the fourier amplitudes
    fourier_amplitudes = np.abs(fourier_image) ** 2

    ## ~~ Constructing a wave vector array ~~ ##
    # what is the wave vector corresponding to
    # an element with indices i and j in the return array?
    # This will automatically return a one dimensional array containing
    # the wave vectors for the numpy.fft.fftn call, in the correct order.
    # By default, the wave vectors are given as a fraction of 1,
    # by multiplying with the total number of pixels, we convert them to a pixel frequency.
    kfreq = np.fft.fftfreq(npix) * npix
    # To convert this to a two dimensional array matching the
    # layout of the two dimensional Fourier image, we can use numpy.meshgrid
    kfreq2D = np.meshgrid(kfreq, kfreq)
    # Finally, we are not really interested in the actual wave vectors, but rather in their norm
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)
    # For what follows, we no longer need the wave vector norms or Fourier image
    # to be laid out as a two dimensional array, so we will flatten them
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    ## ~~ Creating the power spectrum ~~ ##
    # To bin the amplitudes in k space, we need to set up wave number bins.
    # We will create integer k value bins, as is common
    # Note that the maximum wave number will equal half the pixel size of the image.
    # This is because half of the Fourier frequencies can be mapped back to negative wave numbers
    # that have the same norm as their positive counterpart
    kbins = np.arange(0.5, npix // 2 + 1, 1.0)
    # K-values are the midpoint of the bin edges
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    # To compute the average Fourier amplitude (squared) in each bin, we can use scipy.stats.binned_statistic
    Abins, _, _ = binned_statistic(
        knrm, fourier_amplitudes, statistic="mean", bins=kbins
    )
    # Remember that we want the total variance within each bin.
    # Right now, we only have the average power.
    # To get the total power, we need to multiply with the volume in each bin (in 2D, this volume is actually a surface area)
    Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
    return kvals, Abins
