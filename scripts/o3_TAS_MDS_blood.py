import re
import os
import numpy as np
import pandas as pd
import logging
from epilands import generic_read_write, feature_preprocessing, feature_embedding
from epilands.config import ALL_SUBDIRS
from multiprocesstools import MultiProcessHelper
import matplotlib.pyplot as plt
from pathlib import Path
from imagepubautomation.feature_modeling.processes.IO import (
    load_platemap,
    load_h5_feature_data,
    merge_platemap_with_data,
)
from imagepubautomation.feature_modeling.processes.zscore_features import zscore_data

logger = logging.getLogger(__name__)


from config import (
    experiment_name,
    resultsPathRoot,
    label_channels,
    platemap_path,
    sheet_name,
    high_size_threshold,
    low_size_threshold,
    tas_fstring_regex,
    subset,
    cd3_threshold,
)

script_name = Path(__file__).name.removesuffix(".py")

loadPath = resultsPathRoot / experiment_name
assert loadPath.exists(), f"Path {loadPath} does not exist"

print(f"Starting {experiment_name}")

multiprocesshelper = MultiProcessHelper(
    name=script_name,
    working_directory=loadPath,
    loggers=[__name__, *ALL_SUBDIRS],
)

multiprocesshelper.create_directory(script_name)
outputDir = multiprocesshelper.get_directory(script_name)

data = load_h5_feature_data(
    feature_extraction_directory=loadPath / "feature_extraction",
    search_pattern=".hdf5",
)

platemap = load_platemap(
    platemap_path=platemap_path,
    sheet_name=sheet_name,
)

data = merge_platemap_with_data(data=data, platemap=platemap, return_dask=False)
if not data.index.is_unique:
    logger.warning("data index is not unique, dropping")
    data.reset_index(drop=True, inplace=True)
if "OSKM" in experiment_name:
    data.loc[:, "ExperimentalCondition"] = data["ExperimentalCondition"].str.replace(
        "young_i4F_untreated", "Young"
    )
    data.loc[:, "ExperimentalCondition"] = data["ExperimentalCondition"].str.replace(
        "old_i4F_untreated", "Old"
    )
    data.loc[:, "Sample"] = data["Sample"].astype(str)
    data.drop(
        data.index[data["ExperimentalCondition"] == "Mixed_Livers"],
        axis="index",
        inplace=True,
    )

size_inliers = feature_preprocessing.size_threshold_df(
    df=data,
    high_threshold=high_size_threshold,
    low_threshold=low_size_threshold,
    threshold_col="MOR_object_pixel_count",
    threshold_metric="values",
)

zscore = zscore_data(data=data.loc[size_inliers, :], group_by=None, subset=subset)
new = data.columns[~data.columns.isin(zscore.columns)]
zscore[new] = data.loc[size_inliers, new]

fig, ax = plt.subplots()
zscore["CD3_object_average_intensity"].plot.hist(ax=ax, bins=100, range=(0, 500))
plt.vlines(100, ymin=0, ymax=1000)
plt.savefig(outputDir / "CD3_intensity_histogram.png")

cd3_inliers = feature_preprocessing.size_threshold_df(
    df=zscore,
    high_threshold=np.inf,
    low_threshold=cd3_threshold,
    threshold_col="CD3_object_average_intensity",
    threshold_metric="values",
)

zscore.drop(
    zscore.columns[zscore.columns.str.contains("CD3")], axis="columns", inplace=True
)
zscore.loc[:, "CD3_positive"] = False
zscore.loc[cd3_inliers, "CD3_positive"] = True
channels = (
    zscore.columns.str.extractall(re.compile("([a-zA-Z0-9]+)_TXT_TAS"))[0]
    .unique()
    .tolist()
)
channels = channels + ["allchannels"]
data = None
for ch in channels:
    final_file_name = (
        f"mds_embedding_ch{ch}_h{high_size_threshold}_l{low_size_threshold}.csv"
    )
    temp_file_name = final_file_name.replace("csv", "tmp")
    file_not_in_use = multiprocesshelper.create_temp_file(
        final_file_name=final_file_name,
        temp_file_name=temp_file_name,
        path=script_name,
    )
    if file_not_in_use == False:
        logger.debug(f"File {temp_file_name} already exists, skipping...")
        continue
    subset = (
        f"{ch}_TXT_TAS" if ch != "allchannels" else "TXT_TAS"
    )  # Create a new name to save the segmenetation results for this set of images
    try:
        subset = zscore.columns[zscore.columns.str.contains(subset)]
        grouped_data = zscore.groupby(
            ["Sample", "Age", "CD3_positive"], as_index=False
        )[subset].mean()

        fit_data = pd.DataFrame(
            feature_preprocessing.distance_matrix_pdist(grouped_data[subset]),
            index=grouped_data.index,
        )
        fit_data.attrs["name"] = subset
        (
            fit,
            embedding,
            params,
        ) = feature_embedding.generate_mds(
            df_pdist=fit_data,
            metric=True,
            n_components=2,
            save_to=None,
            save_info=False,
        )

        embedding.loc[:, ["Sample", "Age", "CD3_positive"]] = grouped_data[
            ["Sample", "Age", "CD3_positive"]
        ]

        generic_read_write.save_dataframe_to_csv(
            df=embedding,
            path=outputDir,
            filename=f"mds_embedding_ch{ch}_h{high_size_threshold}_l{low_size_threshold}.csv",
        )
        generic_read_write.save_dataframe_to_csv(
            df=params,
            path=outputDir,
            filename=f"mds_params_ch{ch}_h{high_size_threshold}_l{low_size_threshold}.csv",
        )
        multiprocesshelper.delete_tempfile(
            os.path.join(
                multiprocesshelper.get_directory(script_name),
                temp_file_name,
            )
        )
    except Exception as e:
        logger.error(e)
        multiprocesshelper.cleanup()
        if not isinstance(e, KeyError):
            raise e
multiprocesshelper.cleanup()
