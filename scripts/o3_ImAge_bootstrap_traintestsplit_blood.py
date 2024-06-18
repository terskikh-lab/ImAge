import re
import os
import numpy as np
import pandas as pd
import logging
from epilands import generic_read_write, feature_preprocessing
from epilands.config import ALL_SUBDIRS
from multiprocesstools import MultiProcessHelper
from tqdm import tqdm
from random import shuffle
import matplotlib.pyplot as plt
from pathlib import Path
from imagepubautomation.feature_modeling.processes.IO import (
    load_platemap,
    load_h5_feature_data,
    merge_platemap_with_data,
)
from imagepubautomation.feature_modeling.processes.zscore_features import zscore_data
from concurrent.futures import ProcessPoolExecutor, as_completed
from config import (
    experiment_name,
    dataPathRoot,
    resultsPathRoot,
    accuracy_curve_script,
    seed,
    num_iters,
    area_feature,
    tas_fstring_regex,
    group_col,
    sample_col,
    group_A,
    group_B,
    sheet_name,
    subset,
    num_cells,
    num_bootstraps,
    image_script,
    platemap_path,
    cd3_threshold,
    high_size_threshold,
    low_size_threshold,
)

logger = logging.getLogger(__name__)

# Add a date to the image_script name
rng = np.random.default_rng(seed=seed)
script_name = Path(__file__).name.removesuffix(".py")
resultsPath = (resultsPathRoot / experiment_name).resolve()
feature_extraction_directory = (resultsPath / "feature_extraction").resolve()

assert resultsPath.exists(), f"Path {resultsPath} does not exist"

print(f"Starting {experiment_name}")
multiprocesshelper = MultiProcessHelper(
    name=script_name,
    working_directory=resultsPath,
    loggers=[__name__, *ALL_SUBDIRS],
)

multiprocesshelper.create_directory(script_name)
outputDir = multiprocesshelper.get_directory(script_name)

data = load_h5_feature_data(
    feature_extraction_directory=feature_extraction_directory,
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


# Size scatterplot
minmax_range = (0, data[area_feature].quantile(0.98))
bins = int(data[area_feature].quantile(0.98) / 4)
fig, ax = plt.subplots()
data[area_feature].plot.hist(
    bins=bins,
    range=minmax_range,
    ax=ax,
)
plt.vlines(
    [low_size_threshold, high_size_threshold],
    0,
    100,
    colors=["r", "r"],
    linestyles=["--", "--"],
)
plt.title(f"{experiment_name}_size_hist")
plt.savefig(outputDir / "object_size_histogram", dpi=350)
for grp, df in data.groupby(["CD3_object_average_intensity"]):
    fig, ax = plt.subplots()
    df[area_feature].plot.hist(
        bins=bins,
        range=minmax_range,
        ax=ax,
    )
    plt.vlines(
        [low_size_threshold, high_size_threshold],
        0,
        100,
        colors=["r", "r"],
        linestyles=["--", "--"],
    )
    plt.title(f"{experiment_name}_{grp}_size_hist")
    plt.savefig(outputDir / f"{grp}_object_size_histogram", dpi=350)


size_inliers = feature_preprocessing.size_threshold_df(
    df=data,
    high_threshold=high_size_threshold,
    low_threshold=low_size_threshold,
    threshold_col=area_feature,
    threshold_metric="values",
)

data["agenum"] = data["Age"].str.extract("([0-9]+)").astype(int)

cd3_inliers = feature_preprocessing.size_threshold_df(
    df=data,
    high_threshold=np.inf,
    low_threshold=cd3_threshold,
    threshold_col="CD3_object_average_intensity",
    threshold_metric="values",
)

fig, ax = plt.subplots()
data["CD3_object_average_intensity"].plot.hist(ax=ax, bins=100, range=(0, 500))
plt.vlines(100, ymin=0, ymax=1000)
plt.savefig(os.path.join(resultsPath, "CD3_intensity_histogram.png"))

data.drop(data.columns[data.columns.str.contains("CD3")], axis="columns", inplace=True)

channels = (
    data.columns.str.extractall(re.compile(tas_fstring_regex % "([a-zA-Z0-9]+)"))[0]
    .unique()
    .tolist()
)
channels = ["allchannels", *channels]
shuffle(channels)

zscore = zscore_data(data=data.loc[size_inliers, :], group_by=None, subset=subset)
new = data.columns[~data.columns.isin(zscore.columns)]
zscore[new] = data.loc[size_inliers, new]
data = None

for grp in ["PBMC", "CD3+", "CD3-"]:
    if grp == "PBMC":
        grp_inliers = zscore.index
    elif grp == "CD3+":
        grp_inliers = cd3_inliers
    elif grp == "CD3-":
        grp_inliers = ~cd3_inliers
    for ch in channels:
        subset = (
            tas_fstring_regex % ch if ch != "allchannels" else tas_fstring_regex % ""
        )  # Create a new name to save the segmenetation results for this set of images
        try:
            # final_file_name = f"boot_image_{grp}_ch{ch}_h{high_threshold}_l{low_threshold}_ncells{num_cells}_nboots{num_bootstraps}_niters{num_iters}_iter{i}.csv"
            # temp_file_name = final_file_name.replace("csv", "tmp")
            # file_not_in_use = multiprocesshelper.create_temp_file(
            #     final_file_name=final_file_name,
            #     temp_file_name=temp_file_name,
            #     path=script_name,
            # )
            # if file_not_in_use == False:
            #     logger.debug(f"File {temp_file_name} already exists, skipping...")
            #     continue
            # (
            #     boot_image,
            #     boot_image_axis,
            #     boot_train_accuracy,
            #     boot_train_confusion,
            #     boot_test_accuracy,
            #     boot_test_confusion,
            #     boot_accuracy_curve,
            #     group_sizes,
            #     # ) = s1_o1_image_svm_bootstrap_traintestsplit(
            # ) = image_script(
            #     scdata=zscore.loc[grp_inliers, :],
            #     sample_col=sample_col,
            #     group_col=group_col,
            #     group_A=group_A,
            #     group_B=group_B,
            #     num_cells=num_cells,
            #     num_bootstraps=num_bootstraps,
            #     seed=rng.integers(low=0, high=100000),
            #     subset=subset,
            # )
            # multiprocesshelper.delete_tempfile(
            #     os.path.join(
            #         multiprocesshelper.get_directory(script_name),
            #         temp_file_name,
            #     )
            # )

            with ProcessPoolExecutor(max_workers=10) as executor:
                tempfiles_created = []
                futures = []
                for i in np.arange(num_iters):
                    final_file_name = f"boot_image_{grp}_ch{ch}_h{high_size_threshold}_l{low_size_threshold}_ncells{num_cells}_nboots{num_bootstraps}_niters{num_iters}_iter{i}.csv"
                    temp_file_name = final_file_name.replace("csv", "tmp")
                    file_not_in_use = multiprocesshelper.create_temp_file(
                        final_file_name=final_file_name,
                        temp_file_name=temp_file_name,
                        path=script_name,
                    )
                    if file_not_in_use == False:
                        logger.debug(
                            f"File {temp_file_name} already exists, skipping..."
                        )
                        continue
                    tempfiles_created.append(temp_file_name)
                    future = executor.submit(
                        image_script,
                        scdata=zscore.loc[grp_inliers, :],
                        sample_col=sample_col,
                        group_col=group_col,
                        group_A=group_A,
                        group_B=group_B,
                        num_cells=num_cells,
                        num_bootstraps=num_bootstraps,
                        seed=rng.integers(low=0, high=100000),
                        subset=subset,
                    )
                    futures.append(future)
                for i, future in tqdm(
                    enumerate(as_completed(futures)), "Progress on futures"
                ):
                    result = list(future.result())
                    for name, iterResult in zip(
                        [
                            "boot_image",
                            "boot_image_axis",
                            "boot_train_accuracy",
                            "boot_train_confusion",
                            "boot_test_accuracy",
                            "boot_test_confusion",
                            "group_sizes",
                        ],
                        result,
                    ):
                        if not isinstance(iterResult, (pd.DataFrame, pd.Series)):
                            iterResult = pd.Series(iterResult, name=name)
                        generic_read_write.save_dataframe_to_csv(
                            df=iterResult,
                            path=outputDir,
                            filename=tempfiles_created[i]
                            .replace("boot_image", name)
                            .replace("tmp", "csv"),
                        )
                        if name == "boot_image":
                            multiprocesshelper.delete_tempfile(
                                os.path.join(
                                    multiprocesshelper.get_directory(script_name),
                                    tempfiles_created[i],
                                )
                            )
                executor.shutdown(wait=True)

            with ProcessPoolExecutor(max_workers=10) as executor:
                tempfiles_created = []
                acc_futures = []
                for i in np.arange(num_iters):
                    final_file_name = f"boot_accuracy_curve_{grp}_ch{ch}_h{high_size_threshold}_l{low_size_threshold}_ncells{num_cells}_nboots{num_bootstraps}_niters{num_iters}_iter{i}.csv"
                    temp_file_name = final_file_name.replace("csv", "tmp")
                    file_not_in_use = multiprocesshelper.create_temp_file(
                        final_file_name=final_file_name,
                        temp_file_name=temp_file_name,
                        path=script_name,
                    )
                    if file_not_in_use == False:
                        logger.debug(
                            f"File {temp_file_name} already exists, skipping..."
                        )
                        continue
                    tempfiles_created.append(temp_file_name)
                    future = executor.submit(
                        accuracy_curve_script,
                        scdata=zscore.loc[grp_inliers, :],
                        num_bootstraps=num_bootstraps,
                        seed=rng.integers(low=0, high=100000),
                        subset=subset,
                    )
                    acc_futures.append(future)
                for i, future in tqdm(
                    enumerate(as_completed(acc_futures)), "Progress on futures"
                ):
                    iterResult = future.result()
                    generic_read_write.save_dataframe_to_csv(
                        df=iterResult,
                        path=outputDir,
                        filename=tempfiles_created[i].replace("tmp", "csv"),
                    )
                    multiprocesshelper.delete_tempfile(
                        os.path.join(
                            multiprocesshelper.get_directory(script_name),
                            tempfiles_created[i],
                        )
                    )
                executor.shutdown(wait=True)

        except Exception as e:
            logger.error(e)
            multiprocesshelper.cleanup()
            if not isinstance(e, KeyError):
                raise e
    multiprocesshelper.cleanup()
