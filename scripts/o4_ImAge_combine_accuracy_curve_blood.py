import os
import re
import pandas as pd
from pathlib import Path
from config import resultsPathRoot, experiment_name, accuracy_curve_script

meta_pat = "([0-9A-Za-z\+\-]+)_ch([0-9.A-Za-z]+)_h([0-9.A-Za-z]+)_l([0-9]+)_ncells([0-9A-Za-z]+)_niters([0-9]+)_iter([0-9]+)"
mp = meta_pat.replace("(", "").replace(")", "")
file_pat = f"(boot_accuracy_curve_{mp}.csv)"

loadPath = resultsPathRoot / experiment_name / accuracy_curve_script.__name__
savePath = resultsPathRoot / experiment_name / Path(__file__).name.replace(".py", "")
savePath.mkdir(exist_ok=True, parents=True)


files = pd.Series(os.listdir(loadPath))
acc_files = files.str.extract(pat=file_pat).dropna()
acc_files.columns = ["filename"]
params = acc_files["filename"].str.extract(pat=meta_pat)
params.columns = [
    "cell_type",
    "channel",
    "high_threshold",
    "low_threshold",
    "n_cells_per_boot",
    "n_iterations",
    "iteration",
]

metadata = pd.concat([acc_files, params], axis=1)

data = []
for idx, file in enumerate(metadata["filename"]):
    d = pd.read_csv(os.path.join(loadPath, file)).drop(columns="Unnamed: 0")
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
data.head()
data.to_csv(savePath / "boot_accuracy_curve.csv", index=False)
