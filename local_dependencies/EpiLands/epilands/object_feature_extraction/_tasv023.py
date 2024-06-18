import pandas as pd
from typing import Union, List, Dict
import numpy as np
from ._count_pixels_tas import count_pixels_tas


def calculate_TAS_features(object_image_data: Dict[str, np.ndarray]) -> pd.Series:
    TAS_data = [
        extract_MIELv023_tas_features(
            segCellImg=obj_img, mask=object_image_data["masks"], ch=ch
        )
        for ch, obj_img in object_image_data.items()
        if ch != "masks"
    ]
    TAS_data = pd.concat(TAS_data, axis=0)
    return TAS_data


def extract_MIELv023_tas_features(
    segCellImg: np.ndarray,
    mask: np.ndarray,
    ch: Union[str, int],
    percentages: list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
) -> pd.Series:
    """
    Description extract_MIELv023_tas_features
    Generates TAS data as described in ELIFE PAPER

    INPUTS #=====================================
    segCellImg: np.array = raw image to be analyzed
    ch: str = image channel
    percentages: list of floats = percentages used for thresholding. KEEP AS DEFAULT IF REPLICATING ACAPELLA

    OUTPUTS #=====================================
    pd.Series = TAS features as described in acaeplla v2.4

    #================================================
    Martin Alvarez-Kuglen @ Sanford Burnham Prebys Medical Discovery Institute: 2022/01/07
    ADAPTED FROM:
    https://github.com/DWALab/Schormann-et-al/blob/master/MBF_texture_suite_b2.proc.txt

    """
    average_intensity = np.mean(segCellImg, where=mask > 0)
    n_neighbors = 9 if len(segCellImg.shape) == 2 else 27
    # initialize series to store TAS data for each mode (0-3)
    tas_data0 = pd.Series(dtype="float64")
    tas_data1 = pd.Series(dtype="float64")
    tas_data2 = pd.Series(dtype="float64")
    # Extract TAS Features
    for percent in percentages:
        tas_data0 = pd.concat(
            [
                tas_data0,
                pd.Series(
                    data=count_pixels_tas(
                        MIELv023_tas_masking(segCellImg, average_intensity, 0, percent)
                    ),
                    index=MIELv023_tas_name_features(
                        channel=ch,
                        mask_number=0,
                        percent=percent,
                        n_neighbors=n_neighbors,
                    ),
                    dtype="float64",
                ),
            ]
        )
        tas_data1 = pd.concat(
            [
                tas_data1,
                pd.Series(
                    data=count_pixels_tas(
                        MIELv023_tas_masking(segCellImg, average_intensity, 1, percent)
                    ),
                    index=MIELv023_tas_name_features(
                        channel=ch,
                        mask_number=1,
                        percent=percent,
                        n_neighbors=n_neighbors,
                    ),
                    dtype="float64",
                ),
            ]
        )
        tas_data2 = pd.concat(
            [
                tas_data2,
                pd.Series(
                    data=count_pixels_tas(
                        MIELv023_tas_masking(segCellImg, average_intensity, 2, percent)
                    ),
                    index=MIELv023_tas_name_features(
                        channel=ch,
                        mask_number=2,
                        percent=percent,
                        n_neighbors=n_neighbors,
                    ),
                    dtype="float64",
                ),
            ]
        )
    tas_data3 = pd.Series(
        data=count_pixels_tas(
            MIELv023_tas_masking(segCellImg, average_intensity, 3, percent)
        ),
        index=MIELv023_tas_name_features(
            channel=ch,
            mask_number=3,
            percent=None,
            n_neighbors=n_neighbors,
        ),
        dtype="float64",
    )
    return pd.concat([tas_data0, tas_data1, tas_data2, tas_data3])


_masknumber_dict = {
    0: "mean-plus-{x}percent -- mean-minus-{x}percent",
    1: "max -- mean-minus-{x}percent",
    2: "max -- mean-plus-{x}percent",
    3: "max -- mean",
}


def MIELv023_tas_masking(
    image: np.ndarray, mu: Union[float, int], mask_number: int, percentage_number: float
) -> np.ndarray:
    """
    Description MIELv023_tas_masking:
    Generates a masked image according to one of four different thresholding categories:

    0 = (mean+{x}%, mean-{x}%), 1 = (max, mean-{x}%), 2 = (max, mean+{x}%), 3 = (max, mean)

    INPUTS #=====================================

    image: np.array = raw image to be masked

    mu: average pixel intensity for pixels with value > 0 in image

    mask_number: int (0-3) = which mask option to use (see above for options)

    percentage_number: float = percentage used when thresholding for mask

    OUTPUTS #=====================================

    np.array = new image masked by the threshold values given above.

    #================================================

    Martin Alvarez-Kuglen @ Sanford Burnham Prebys Medical Discovery Institute: 2022/01/07

    ADAPTED FROM Mahotas package v: mahotas\features\tas.py
    """

    if mask_number == 0:
        maximum, minimum = (1 + percentage_number, 1 - percentage_number)
        mask1 = np.where(image < minimum * mu, 0, 1)
        mask2 = np.where(image < maximum * mu, 0, 1)
    if mask_number == 1:
        minimum = 1 - percentage_number
        mask1 = np.where(image < minimum * mu, 0, 1)
        mask2 = np.zeros_like(image)
    if mask_number == 2:
        minimum = 1 + percentage_number
        mask1 = np.where(image < minimum * mu, 0, 1)
        mask2 = np.zeros_like(image)
    if mask_number == 3:
        mask1 = np.where(image < mu, 0, 1)
        mask2 = np.zeros_like(image)

    newImg = np.subtract(mask1, mask2)
    return newImg


def MIELv023_tas_name_features(
    channel: Union[str, int],
    mask_number: int,
    percent: float,
    original_names: str = True,
    n_neighbors: int = 9,
) -> List[str]:
    """
    Description MIELv023_tas_name_features:
    Generates a list of 9 strings (one for each TAS statistic, ie, N-neighbors counted)
    INPUTS #=====================================
    channel: str = channel name. EX: Dapi
    mask_number: int in range (0,3) = kind of threshold used. See _masknumber_dict for naming.
    percent: float = percentage used when thresholding for mask (see x above). Use None if mask_number = 3
    OUTPUTS #=====================================
    list = 9 strings (one for each TAS statistic, ie, N-neighbors counted)
    #================================================
    Martin Alvarez-Kuglen @ Sanford Burnham Prebys Medical Discovery Institute: 2022/01/07
    ADAPTED FROM Mahotas package v: mahotas\features\tas.py
    """
    if original_names:
        if mask_number == 3:
            names = [
                "_".join((str(channel), "TXT", "TAS", "{}")).format(i + 28)
                for i in range(n_neighbors)
            ]
        else:
            names = [
                "_".join(
                    (str(channel), "TXT", "TAS", "{}", str(int(percent * 100)))
                ).format(i + n_neighbors * mask_number + 1)
                for i in range(n_neighbors)
            ]

    else:
        if percent == None:
            names = [
                "_".join(
                    (
                        str(channel),
                        "TXT",
                        "TAS",
                        "{}neighbors",
                        _masknumber_dict[mask_number],
                    )
                ).format(i)
                for i in range(n_neighbors)
            ]
        else:
            names = [
                "_".join(
                    (
                        str(channel),
                        "TXT",
                        "TAS",
                        "{}neighbors",
                        _masknumber_dict[mask_number].format_map(
                            {"x": int(percent * 100)}
                        ),
                    )
                ).format(i)
                for i in range(n_neighbors)
            ]
    return names
