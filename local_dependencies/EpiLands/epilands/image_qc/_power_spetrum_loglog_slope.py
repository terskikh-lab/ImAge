import logging
import numpy as np
from scipy.stats import linregress
from typing import Union
import os

from ._power_spectrum import power_spectrum

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)

## Image Quality Control


def power_spectrum_loglog_slope(
    image: np.ndarray, plot_result: bool = True
) -> Union[int, float]:
    logger.info("Calculating power spectrum")
    kvals, Abins = power_spectrum(image)
    slope, intercept, r_value, p_value, std_err = linregress(
        np.log10(kvals), np.log10(Abins)
    )
    logger.info(f"Power spectrum slope: {slope}")
    # if plot_result == True:
    #     fig = plt.figure()
    #     xfid = np.linspace(0,3)     # This is just a set of x to plot the straight line
    #     plt.plot(np.log10(kvals),
    #              np.log10(Abins),
    #              'k.')
    #     plt.plot(xfid,
    #              xfid*slope+intercept,
    #              label='log(K) = {:.2f}*log(P) + {:.2f}'.format(slope, intercept))
    #     plt.title('Power spectrum')
    #     plt.xlabel('Log(P(k))')
    #     plt.ylabel('Log(k)')
    #     plt.legend()
    #        plt.show()
    return slope
