import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from config import resultsPathRoot, experiment_name, image_script

loadPath = resultsPathRoot / experiment_name / image_script.__name__
savePath = resultsPathRoot / experiment_name / Path(__file__).name.removesuffix(".py")

metapat = "([0-9.A-Za-z+-]+)_ch([0-9.A-Za-z]+)_h([0-9.A-Za-z]+)_l([0-9.]+)_ncells([0-9.]+)_nboots([0-9.]+)_niters([0-9.]+)_iter([0-9]+)"
mp = metapat.replace("(", "").replace(")", "")
filepat = f"(boot_image_{mp}.csv)"
files = pd.Series(os.listdir(loadPath))
acc_files = files.str.extract(pat=filepat).dropna()

params = acc_files[0].str.extract(pat=metapat)
acc_files.columns = ["filename"]
params.columns = [
    "subset",
    "channel",
    "high_threshold",
    "low_threshold",
    "n_cells_per_boot",
    "n_bootstraps",
    "n_iterations",
    "iteration",
]

metadata = pd.concat([acc_files, params], axis=1)
data = []
for idx, file in tqdm(enumerate(metadata["filename"])):
    d = pd.read_csv(os.path.join(loadPath, file)).drop(columns="Unnamed: 0")
    d_grouped = d.groupby(["split", "Sample", "agenum", "ExperimentalCondition"])
    d = pd.concat([d_grouped.mean(), d_grouped.std()], axis=1)

    ncols = len(d.columns)
    newcols = [
        *(d.columns[: int(ncols / 2)] + "_mean"),
        *(d.columns[int(ncols / 2) :] + "_std"),
    ]
    d.columns = newcols
    d["filename"] = metadata["filename"].iloc[idx]
    d["channel"] = metadata["channel"].iloc[idx]
    data.append(d)
data = pd.concat(data, axis=0)
data = pd.merge(data.reset_index(), metadata, on=["filename", "channel"])
data["dataset"] = experiment_name
if "0" in data["Sample"].values:
    data = data[~(data["Sample"] == "0")]

data.to_csv(savePath / "boot_image.csv", index=False)
