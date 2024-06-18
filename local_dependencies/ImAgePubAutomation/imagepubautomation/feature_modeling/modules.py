import pandas as pd
import numpy as np
from epilands.image_model import ImAgeModel
import logging
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC


logger = logging.getLogger("modules")


def s1_o1_image_cenvec_bootstrap_traintestsplit(
    scdata,
    sample_col,
    group_col,
    group_A,
    group_B,
    num_cells,
    num_bootstraps,
    seed,
    subset: str,
):
    print(f"starting image_with_accuracy using seed: {seed}")
    # scdata = scdata.groupby(["Sample", "ExperimentalCondition"]).sample(200)

    # num_bootstraps = 100
    test_size = 0.25
    train_size = 0.75

    # boot_accuracy_curve = s1_o1_image_cenvec_bootstrap_accuracy_curve(
    #     scdata=scdata,
    #     num_bootstraps=num_bootstraps,
    #     seed=seed,
    #     subset=subset,
    # )

    # subset = "TXT_TAS"
    relevant_features = scdata.columns[scdata.columns.str.contains(subset)]

    data_grouped = scdata.groupby(["Sample", "ExperimentalCondition"])
    train_data = []
    test_data = []
    for grp, dat in data_grouped:
        grp_train_data, grp_test_data = train_test_split(
            dat.index.values,
            test_size=test_size,
            train_size=train_size,
            random_state=seed,
        )
        train_data.append(grp_train_data)
        test_data.append(grp_test_data)
    grp_train_data = None
    grp_test_data = None
    train_data = np.concatenate(train_data)
    test_data = np.concatenate(test_data)

    scdata.loc[test_data, "split"] = "test"
    scdata.loc[train_data, "split"] = "train"
    assert not any(scdata["split"].isna().values)
    data_grouped = scdata.groupby(
        ["split", "Sample", "agenum", "ExperimentalCondition"],
        as_index=True,
    )
    data_groups = list(data_grouped.groups.keys())
    if len(data_grouped) % 2 != 0:  # add nnonzero
        raise ValueError("Train test split dropped a group")
    # if any([i < num_bootstraps for i in test_grouped.size().values]):
    #     raise ValueError("Not enough cells to continue")
    scdata = None

    rng = np.random.Generator(np.random.PCG64(seed=seed))
    # start = time.perf_counter()
    group_sizes = data_grouped.size()
    boot_data = []
    boot_meta = []
    for group, size in tqdm([*group_sizes.items()], "bootstrapping"):
        if num_cells == "original":
            num_cells = size
        boot_idxs = rng.choice(size, size=(num_bootstraps, num_cells), replace=True)
        temp_data = data_grouped.get_group(group)[relevant_features]
        for i, idxs in enumerate(boot_idxs):
            boot_data.append(temp_data.iloc[idxs].mean().values)
            boot_meta.append(np.array([i + 1, *group]))

    boot_meta = pd.DataFrame(
        np.array(boot_meta),
        columns=[
            "bootstrap",
            "split",
            "Sample",
            "agenum",
            "ExperimentalCondition",
        ],
    )
    boot_data = pd.DataFrame(
        np.array(boot_data),
        columns=relevant_features,
    )
    boot_data = pd.concat([boot_meta, boot_data], axis=1)
    # stop = time.perf_counter()
    # print((stop - start) / 60)

    image_model = ImAgeModel()
    image_model.fit(
        data=boot_data.loc[boot_data["split"] == "train"][
            [sample_col, group_col, *relevant_features]
        ],
        sample_col=sample_col,
        group_col=group_col,
        feature_cols=relevant_features,
        group_A=group_A,
        group_B=group_B,
    )
    boot_image_axis = image_model.coef_

    # Full Data
    boot_image = image_model.score(
        data=boot_data[[*relevant_features]],
    ).to_frame()
    boot_image.loc[:, "ImAge_orthogonal_distance"] = image_model.score_orthogonal(
        data=boot_data[[*relevant_features]],
    )
    boot_image[["split", "Sample", "agenum", "ExperimentalCondition"]] = boot_data[
        ["split", "Sample", "agenum", "ExperimentalCondition"]
    ]
    image_grouped = boot_image.groupby(["split", "ExperimentalCondition"])
    # Get train accuracy
    yo_score = pd.concat(
        [
            image_grouped.get_group(("train", "Young")),
            image_grouped.get_group(("train", "Old")),
        ]
    )
    yo_pred = yo_score["ImAge"] > image_model.threshold
    yo_true = yo_score["ExperimentalCondition"] == "Old"
    boot_train_accuracy = accuracy_score(yo_true, yo_pred)
    boot_train_confusion = pd.Series(
        data=confusion_matrix(yo_true, yo_pred).ravel(),
        index=["true_neg", "false_pos", "false_neg", "true_pos"],
    )

    yo_score = pd.concat(
        [
            image_grouped.get_group(("test", "Young")),
            image_grouped.get_group(("test", "Old")),
        ]
    )
    yo_pred = yo_score["ImAge"] > image_model.threshold
    yo_true = yo_score["ExperimentalCondition"] == "Old"
    boot_test_accuracy = accuracy_score(yo_true, yo_pred)
    boot_test_confusion = pd.Series(
        data=confusion_matrix(yo_true, yo_pred).ravel(),
        index=["true_neg", "false_pos", "false_neg", "true_pos"],
    )

    return (
        boot_image,
        boot_image_axis,
        boot_train_accuracy,
        boot_train_confusion,
        boot_test_accuracy,
        boot_test_confusion,
        # boot_accuracy_curve,
        group_sizes,
    )


def s1_o1_image_cenvec_bootstrap_accuracy_curve(
    scdata,
    num_bootstraps,
    seed,
    subset: str,
):
    # num_bootstraps = 100
    test_size = 0.25
    train_size = 0.75

    # subset = "TXT_TAS"
    relevant_features = scdata.columns[scdata.columns.str.contains(subset)]

    # zscore = zscore_data(data=scdata, group_by=None, subset=subset)
    # new = scdata.columns[~scdata.columns.isin(zscore.columns)]
    # zscore = pd.concat([zscore, scdata.loc[:, new]], axis=1)
    # scdata = zscore

    data_grouped = scdata.groupby(["Sample", "ExperimentalCondition"])
    train_data = []
    test_data = []
    for grp, dat in data_grouped:
        grp_train_data, grp_test_data = train_test_split(
            dat.index.values,
            test_size=test_size,
            train_size=train_size,
            random_state=seed,
        )
        train_data.append(grp_train_data)
        test_data.append(grp_test_data)
    grp_train_data = None
    grp_test_data = None
    train_data = np.concatenate(train_data)
    test_data = np.concatenate(test_data)

    scdata.loc[test_data, "split"] = "test"
    scdata.loc[train_data, "split"] = "train"
    data_grouped = scdata.groupby(
        ["split", "Sample", "agenum", "ExperimentalCondition"],
        as_index=True,
    )
    data_groups = list(data_grouped.groups.keys())
    if len(data_grouped) % 2 != 0:  # if nonzero
        raise ValueError("Train test split dropped a group")
    # if any([i < num_bootstraps for i in test_grouped.size().values]):
    #     raise ValueError("Not enough cells to continue")
    scdata = None

    rng = np.random.Generator(np.random.PCG64(seed=seed))

    acc_curve = []
    group_sizes = data_grouped.size()
    max_size = group_sizes.min()
    acc_bin_sizes = [
        1,
        *np.logspace(1, np.log10(max_size), base=10, num=25).astype(int),
    ]
    # start = time.perf_counter()
    for s in tqdm(acc_bin_sizes, "accuracy curve"):
        boot_data = []
        boot_meta = []
        for group, size in group_sizes.items():
            if not any(i in ["Young", "Old"] for i in group):
                continue
            if s == "original":
                s = size
            boot_idxs = rng.choice(size, size=(num_bootstraps, s), replace=True)
            temp_data = data_grouped.get_group(group)
            for i, idxs in enumerate(boot_idxs):
                temp_boot = temp_data.iloc[idxs][relevant_features].mean()
                boot_data.append(temp_boot)
            boot_meta.append(
                pd.DataFrame(
                    np.array([*group] * num_bootstraps).reshape(
                        (num_bootstraps, len(group))
                    ),
                    columns=["split", "Sample", "agenum", "ExperimentalCondition"],
                )
            )
        boot_data = pd.concat(boot_data, ignore_index=True, axis=1).T
        boot_meta = pd.concat(boot_meta, ignore_index=True, axis=0)
        boot_data = pd.concat([boot_meta, boot_data], axis=1)
        for_training = boot_data["split"] == "train"

        image_model = ImAgeModel()
        image_model.fit(
            data=boot_data.loc[
                for_training, ["Sample", "ExperimentalCondition", *relevant_features]
            ],
            sample_col="Sample",
            group_col="ExperimentalCondition",
            feature_cols=relevant_features,
            group_A="Young",
            group_B="Old",
        )
        boot_image = image_model.score(
            data=boot_data.loc[~for_training, relevant_features],
        ).to_frame()
        boot_image["ExperimentalCondition"] = boot_data.loc[
            ~for_training, "ExperimentalCondition"
        ]
        boot_yo_pred = boot_image["ImAge"] > image_model.threshold
        boot_yo_true = boot_image["ExperimentalCondition"] == "Old"
        acc_curve.append(accuracy_score(boot_yo_true, boot_yo_pred))
    # stop = time.perf_counter()
    # print((stop - start) / 60)
    boot_accuracy_curve = pd.DataFrame(
        [acc_curve, acc_bin_sizes], index=["accuracy", "bin_size"]
    ).T

    return boot_accuracy_curve


def s1_o1_image_svm_bootstrap_traintestsplit(
    scdata,
    sample_col,
    group_col,
    group_A,
    group_B,
    num_cells,
    num_bootstraps,
    seed,
    subset: str,
):
    print(f"starting image using seed: {seed}")
    # scdata = scdata.groupby(["Sample", "ExperimentalCondition"]).sample(200)

    # num_bootstraps = 100
    test_size = 0.25
    train_size = 0.75

    # subset = "TXT_TAS"
    relevant_features = scdata.columns[scdata.columns.str.contains(subset)]

    # zscore = zscore_data(data=scdata, group_by=None, subset=subset)
    # new = scdata.columns[~scdata.columns.isin(zscore.columns)]
    # zscore = pd.concat([zscore, scdata.loc[:, new]], axis=1)
    # scdata = zscore

    # boot_accuracy_curve = s1_o1_image_svm_bootstrap_accuracy_curve(
    #     scdata=scdata,
    #     num_bootstraps=num_bootstraps,
    #     seed=seed,
    #     subset=subset,
    # )

    data_grouped = scdata.groupby(["Sample", "ExperimentalCondition"])
    train_data = []
    test_data = []
    for grp, dat in data_grouped:
        grp_train_data, grp_test_data = train_test_split(
            dat.index.values,
            test_size=test_size,
            train_size=train_size,
            random_state=seed,
        )
        train_data.append(grp_train_data)
        test_data.append(grp_test_data)
    grp_train_data = None
    grp_test_data = None
    train_data = np.concatenate(train_data)
    test_data = np.concatenate(test_data)

    scdata.loc[test_data, "split"] = "test"
    scdata.loc[train_data, "split"] = "train"
    assert not any(scdata["split"].isna().values)
    data_grouped = scdata.groupby(
        ["split", "Sample", "agenum", "ExperimentalCondition"],
        as_index=True,
    )
    scdata = None
    data_groups = list(data_grouped.groups.keys())
    if len(data_grouped) % 2 != 0:  # add nnonzero
        raise ValueError("Train test split dropped a group")
    # if any([i < num_bootstraps for i in test_grouped.size().values]):
    #     raise ValueError("Not enough cells to continue")

    rng = np.random.Generator(np.random.PCG64(seed=seed))
    # start = time.perf_counter()
    group_sizes = data_grouped.size()
    boot_data = []
    boot_meta = []
    for group, size in tqdm([*group_sizes.items()], "bootstrapping"):
        if num_cells == "original":
            num_cells = size
        boot_idxs = rng.choice(size, size=(num_bootstraps, num_cells), replace=True)
        temp_data = data_grouped.get_group(group)[relevant_features]
        for i, idxs in enumerate(boot_idxs):
            boot_data.append(temp_data.iloc[idxs].mean().values)
            boot_meta.append(np.array([i + 1, *group]))

    boot_meta = pd.DataFrame(
        np.array(boot_meta),
        columns=[
            "bootstrap",
            "split",
            "Sample",
            "agenum",
            "ExperimentalCondition",
        ],
    )
    boot_data = pd.DataFrame(
        np.array(boot_data),
        columns=relevant_features,
    )
    boot_data = pd.concat([boot_meta, boot_data], axis=1)
    # stop = time.perf_counter()
    # print((stop - start) / 60)

    image_model = SVC(
        kernel="linear", verbose=True, class_weight="balanced", probability=False
    )
    for_training = (boot_data["split"] == "train") & boot_data[
        "ExperimentalCondition"
    ].isin([group_A, group_B])
    image_model.fit(
        X=boot_data.loc[for_training, relevant_features],
        y=boot_data.loc[for_training, group_col] == group_B,
    )
    # Get feature weights
    boot_image_axis = pd.Series(
        data=image_model.coef_.squeeze(),
        index=image_model.feature_names_in_,
    )
    # Full Data
    boot_image = pd.DataFrame(
        image_model.decision_function(X=boot_data[[*relevant_features]]),
        index=boot_data.index,
        columns=["ImAge"],
    )
    # Get the hyperplane coefficients and intercept, normalize
    coef_ = image_model.coef_.squeeze()
    coef_l2norm = np.linalg.norm(coef_, ord=2)
    coef_unit = coef_ / coef_l2norm
    b = image_model.intercept_

    def _scalar_projection(data):
        return np.dot(data, coef_) / coef_l2norm

    def _vector_projection(data):
        return np.outer(_scalar_projection(data), coef_unit)

    def _ortho_vector(data):
        return data - _vector_projection(data)

    def _ortho_distance(data):
        return np.linalg.norm(_ortho_vector(data), ord=2)

    # Center plane at 0
    data = boot_data[[*relevant_features]] - b[0]
    # project data onto plane via orthogonal projection
    # calculate centroid
    data_plane_centroid = _ortho_vector(data).mean(axis=0)
    # Center the data at the centroid
    data -= data_plane_centroid
    # Find the distance to the normal of the hyperplane
    image_ortho_distance = _ortho_vector(data).apply(np.linalg.norm, axis=1)
    # add to dataframe
    boot_image["image_orthogonal_distance"] = image_ortho_distance
    boot_image["image_distance"] = boot_image["ImAge"] / coef_l2norm

    boot_image[["split", "Sample", "agenum", "ExperimentalCondition"]] = boot_data[
        ["split", "Sample", "agenum", "ExperimentalCondition"]
    ]
    image_grouped = boot_image.groupby(["split", "ExperimentalCondition"])
    # Get train accuracy
    yo_score = pd.concat(
        [
            image_grouped.get_group(("train", group_A)),
            image_grouped.get_group(("train", group_B)),
        ]
    )
    yo_pred = yo_score["ImAge"] > 0
    yo_true = yo_score["ExperimentalCondition"] == group_B
    boot_train_accuracy = accuracy_score(yo_true, yo_pred)
    boot_train_confusion = pd.Series(
        data=confusion_matrix(yo_true, yo_pred).ravel(),
        index=["true_neg", "false_pos", "false_neg", "true_pos"],
    )

    yo_score = pd.concat(
        [
            image_grouped.get_group(("test", group_A)),
            image_grouped.get_group(("test", group_B)),
        ]
    )
    yo_pred = yo_score["ImAge"] > 0
    yo_true = yo_score["ExperimentalCondition"] == group_B
    boot_test_accuracy = accuracy_score(yo_true, yo_pred)
    boot_test_confusion = pd.Series(
        data=confusion_matrix(yo_true, yo_pred).ravel(),
        index=["true_neg", "false_pos", "false_neg", "true_pos"],
    )

    return (
        boot_image,
        boot_image_axis,
        boot_train_accuracy,
        boot_train_confusion,
        boot_test_accuracy,
        boot_test_confusion,
        # boot_accuracy_curve,
        group_sizes,
    )


def s1_o1_image_svm_bootstrap_accuracy_curve(
    scdata,
    num_bootstraps,
    seed,
    subset: str,
):
    # num_bootstraps = 100
    test_size = 0.25
    train_size = 0.75

    # subset = "TXT_TAS"
    relevant_features = scdata.columns[scdata.columns.str.contains(subset)]

    # zscore = zscore_data(data=scdata, group_by=None, subset=subset)
    # new = scdata.columns[~scdata.columns.isin(zscore.columns)]
    # zscore = pd.concat([zscore, scdata.loc[:, new]], axis=1)
    # scdata = zscore

    data_grouped = scdata.groupby(["Sample", "ExperimentalCondition"])
    train_data = []
    test_data = []
    for grp, dat in data_grouped:
        grp_train_data, grp_test_data = train_test_split(
            dat.index.values,
            test_size=test_size,
            train_size=train_size,
            random_state=seed,
        )
        train_data.append(grp_train_data)
        test_data.append(grp_test_data)
    grp_train_data = None
    grp_test_data = None
    train_data = np.concatenate(train_data)
    test_data = np.concatenate(test_data)

    scdata.loc[test_data, "split"] = "test"
    scdata.loc[train_data, "split"] = "train"
    data_grouped = scdata.groupby(
        ["split", "Sample", "agenum", "ExperimentalCondition"],
        as_index=True,
    )
    data_groups = list(data_grouped.groups.keys())
    if len(data_grouped) % 2 != 0:  # add nnonzero
        raise ValueError("Train test split dropped a group")
    # if any([i < num_bootstraps for i in test_grouped.size().values]):
    #     raise ValueError("Not enough cells to continue")
    scdata = None

    rng = np.random.Generator(np.random.PCG64(seed=seed))

    acc_curve = []
    group_sizes = data_grouped.size()
    max_size = group_sizes.min()
    acc_bin_sizes = [
        1,
        *np.logspace(1, np.log10(max_size), base=10, num=25).astype(int),
    ]
    # start = time.perf_counter()
    for s in tqdm(acc_bin_sizes, "accuracy curve"):
        boot_data = []
        boot_meta = []
        for group, size in group_sizes.items():
            if not any(i in ["Young", "Old"] for i in group):
                continue
            if s == "original":
                s = size
            boot_idxs = rng.choice(size, size=(num_bootstraps, s), replace=True)
            temp_data = data_grouped.get_group(group)
            for i, idxs in enumerate(boot_idxs):
                temp_boot = temp_data.iloc[idxs][relevant_features].mean()
                boot_data.append(temp_boot)
            boot_meta.append(
                pd.DataFrame(
                    np.array([*group] * num_bootstraps).reshape(
                        (num_bootstraps, len(group))
                    ),
                    columns=["split", "Sample", "agenum", "ExperimentalCondition"],
                )
            )
        boot_data = pd.concat(boot_data, ignore_index=True, axis=1).T
        boot_meta = pd.concat(boot_meta, ignore_index=True, axis=0)
        boot_data = pd.concat([boot_meta, boot_data], axis=1)
        imagemodel = SVC(
            kernel="linear", verbose=True, class_weight="balanced", probability=False
        )
        for_training = boot_data["split"] == "train"
        imagemodel.fit(
            X=boot_data.loc[for_training, relevant_features],
            y=boot_data.loc[for_training, "ExperimentalCondition"] == "Old",
        )
        boot_image = pd.DataFrame(
            imagemodel.decision_function(
                X=boot_data.loc[~for_training, relevant_features]
            ),
            index=boot_data.loc[~for_training, :].index,
            columns=["ImAge"],
        )
        boot_image["ExperimentalCondition"] = boot_data.loc[
            ~for_training, "ExperimentalCondition"
        ]
        boot_yo_pred = boot_image["ImAge"] > 0
        boot_yo_true = boot_image["ExperimentalCondition"] == "Old"
        acc_curve.append(accuracy_score(boot_yo_true, boot_yo_pred))
    # stop = time.perf_counter()
    # print((stop - start) / 60)
    boot_accuracy_curve = pd.DataFrame(
        [acc_curve, acc_bin_sizes], index=["accuracy", "bin_size"]
    ).T

    return boot_accuracy_curve
