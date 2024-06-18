import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass
from skimage.measure import moments
from typing import Dict


from scipy.spatial import distance
from skimage.measure import regionprops
import numpy as np


def object_pixel_count(image_data: Dict[str, np.ndarray]) -> pd.Series:
    object_features = {f"MOR_object_pixel_count": np.count_nonzero(image_data["masks"])}
    return pd.Series(object_features)


def perimeter(mask):
    """
    Calculate the perimeter of a 2D object represented by a binary mask.

    Parameters:
    mask (numpy.ndarray): A binary mask of the object.

    Returns:
    float: The perimeter of the object.
    """
    # Calculate the perimeter of the mask
    perimeter = np.sum(mask[:, 1:] != mask[:, :-1]) + np.sum(
        mask[1:, :] != mask[:-1, :]
    )
    return perimeter


def roundness(mask):
    """
    Calculates the roundness of a binary mask of a numpy array.

    Parameters:
    -----------
    mask : numpy.ndarray
        A binary mask of a NumPy array.

    Returns:
    --------
    float
        The roundness of the binary mask.
    """

    # Calculate the perimeter of the mask
    perim = perimeter(mask)

    # Calculate the area of the mask
    area = np.sum(mask)

    # Calculate the roundness of the mask
    roundness = (4 * np.pi * area) / (perim**2)

    return roundness


def eccentricity(mask):
    """
    Calculates the eccentricity of a binary mask of a numpy array.

    Parameters:
    -----------
    mask : numpy.ndarray
        A binary mask of a NumPy array.

    Returns:
    --------
    float
        The eccentricity of the binary mask.
    """

    # Calculate the moments of the mask
    m = moments(mask)

    # Calculate the centroid of the mask
    cy, cx = center_of_mass(mask)

    # Calculate the eccentricity of the mask
    eccentricity = (
        np.sqrt(1 - ((m[2, 0] - m[0, 2]) ** 2 / ((m[2, 0] + m[0, 2]) ** 2)))
        * (m[0, 0] / m[1, 1])
        * np.sqrt(
            (m[2, 0] + m[0, 2] + np.sqrt((m[2, 0] - m[0, 2]) ** 2 + 4 * m[1, 1] ** 2))
            / (m[2, 0] + m[0, 2] - np.sqrt((m[2, 0] - m[0, 2]) ** 2 + 4 * m[1, 1] ** 2))
        )
    )

    return eccentricity


def sphericity(mask):
    """
    Calculate the sphericity of a 2D object represented by a binary mask.

    Parameters:
    mask (numpy.ndarray): A binary mask of the object.

    Returns:
    float: The sphericity of the object.
    """
    area = np.sum(mask)
    perim = perimeter(mask)
    sphericity = (np.pi ** (1 / 3) * (6 * area) ** (2 / 3)) / perim
    return sphericity


def spherical_disproportion(mask):
    """
    Calculate the spherical disproportion of a 2D object represented by a binary mask.

    Parameters:
    mask (numpy.ndarray): A binary mask of the object.

    Returns:
    float: The spherical disproportion of the object.
    """
    return 1 / sphericity(mask)


def max_2d_diameter(mask):
    """
    Calculate the maximum 2D diameter of a 2D object represented by a binary mask.

    Parameters:
    mask (numpy.ndarray): A binary mask of the object.

    Returns:
    float: The maximum 2D diameter of the object.
    """
    points = np.argwhere(mask)  # get the coordinates of all non-zero points
    distances = distance.pdist(points, "euclidean")  # calculate all pairwise distances
    max_diameter = np.max(distances)
    return max_diameter


def major_minor_axis_length(mask):
    """
    Calculate the major and minor axis lengths of a 2D object represented by a binary mask.

    Parameters:
    mask (numpy.ndarray): A binary mask of the object.

    Returns:
    tuple: The major and minor axis lengths of the object.
    """
    props = regionprops(mask.astype(int))[0]  # calculate properties of the mask
    major_axis_length = props.major_axis_length
    minor_axis_length = props.minor_axis_length
    return major_axis_length, minor_axis_length


def elongation(mask):
    """
    Calculate the elongation of a 2D object represented by a binary mask.

    Parameters:
    mask (numpy.ndarray): A binary mask of the object.

    Returns:
    float: The elongation of the object.
    """
    major_axis_length, minor_axis_length = major_minor_axis_length(mask)
    elongation = major_axis_length / minor_axis_length
    return elongation
