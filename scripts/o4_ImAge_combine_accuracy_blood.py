import os
import re
import pandas as pd
from pathlib import Path
from config import experiment_name, resultsPathRoot, image_script


loadPath = resultsPathRoot / experiment_name / image_script.__name__
savePath = resultsPathRoot / experiment_name / Path(__file__).name.replace(".py", "")
savePath.mkdir(exist_ok=True, parents=True)

files = pd.Series(os.listdir(loadPath))
acc_files = files.str.extract(
    pat="(boot_[a-zA-Z]+_accuracy_[0-9A-Za-z\+\-_.]+.csv)"
).dropna()
acc_files.columns = ["filename"]
params = acc_files["filename"].str.extract(
    pat="([0-9A-Za-z\+\-]+)_ch([0-9A-Za-z]+)_h([0-9A-Za-z]+)_l([0-9]+)_ncells([0-9A-Za-z]+)_nboots([0-9A-Za-z]+)_niters([0-9]+)_iter([0-9]+)"
)
params.columns = [
    "cell_type",
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
for idx, file in enumerate(metadata["filename"]):
    d = pd.read_csv(os.path.join(loadPath, file)).drop(columns="Unnamed: 0")
    split = d.columns.str.extract("boot_([A-Za-z]+)_accuracy")[0][0]
    cols = d.columns.str.replace(f"_{split}", "")
    d.columns = cols
    d["split"] = split
    d["filename"] = file
    data.append(d)
data = pd.merge(pd.concat(data, axis=0, ignore_index=True), metadata, on="filename")
data["dataset"] = experiment_name
data.sort_values(by=["channel"], ascending=False, inplace=True)
if ("k27" in experiment_name) or ("k4" in experiment_name):
    data["experiment"] = re.findall(
        re.compile("([A-Za-z0-9_]+)_k[A-Za-z0-9]+"), experiment_name
    )[0]
else:
    data["experiment"] = experiment_name
data.to_csv(savePath / f"{experiment_name}_boot_accuracy.csv", index=False)
