from __future__ import annotations

# Import libraries
import pandas as pd
import numpy as np
from .tools import join_iterable
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix

# Relative imports


class ImAgeModel:
    def __init__(self) -> None:
        self.A_centroid: pd.Series = None
        self.B_centroid: pd.Series = None
        self.ImAgeVec: pd.Series = None
        self.l2_norm: float = None
        self.coef_: pd.Series = None
        self.feature_names_in_: list = None
        self.n_features_in_: int = None
        self.classes_: list = None
        self.scores: pd.Series = None
        self.auc: float = None
        self.threshold: float = None
        self.accuracy_score: float = None
        self.confusion_matrix: pd.Series = None

    def fit(
        self,
        data: pd.DataFrame,
        sample_col: str,
        group_col: str,
        feature_cols,
        group_A: str,
        group_B: str,
    ):
        data = data.set_index([sample_col, group_col])
        sample_means = data.groupby([sample_col, group_col])[feature_cols].mean()
        groups = sample_means.groupby(group_col, as_index=True)
        A_centroid = groups.get_group(group_A).mean()
        A_centroid.attrs["name"] = f"{join_iterable(group_A)} centroid"
        B_centroid = groups.get_group(group_B).mean()
        B_centroid.attrs["name"] = f"{join_iterable(group_B)} centroid"

        self.A_centroid = A_centroid
        self.B_centroid = B_centroid
        self.ImAgeVec = B_centroid - A_centroid
        self.l2_norm = np.linalg.norm(self.ImAgeVec.values, ord=2)

        self.coef_ = self.ImAgeVec / self.l2_norm
        self.feature_names_in_ = data.columns
        self.n_features_in_ = len(self.feature_names_in_)
        self.classes_ = [group_A, group_B]

        self.scores = self.score(data)
        refidx = np.logical_or(
            data.index.get_level_values(1) == group_A,
            data.index.get_level_values(1) == group_B,
        )
        reflabels = data[refidx].index.get_level_values(1) == group_B
        refscores = self.scores[refidx]
        # labels = data.index.to_series()[refidx].apply(lambda idx: group_B in idx)
        self._roc_auc_analysis(refscores, reflabels)

    def _scalar_projection(self, data_vec):
        # return np.dot(data_vec, self.ImAgeVec) / self.l2_norm
        return np.dot((data_vec - self.A_centroid), self.ImAgeVec) / self.l2_norm

    def _vector_projection(self, data_vec):
        return np.dot(self._scalar_projection(data_vec), self.ImAgeVec / self.l2_norm)

    def _ortho_projection(self, data_vec):
        return data_vec - self.A_centroid - self._vector_projection(data_vec)

    def _ortho_distance(self, data_vec):
        # return np.linalg.norm(
        #     data_vec - self._vector_projection(data_vec), ord=2
        # )
        return np.linalg.norm(self._ortho_projection(data_vec), ord=2)

    def _roc_auc_analysis(self, scores, labels):
        y_true = labels
        y_score = scores.values.reshape(-1, 1)
        fpr, tpr, thresholds = roc_curve(
            y_true=y_true,
            y_score=y_score,
        )
        auc = roc_auc_score(y_true, y_score)
        if auc < 1:
            threshold = thresholds[np.argmin((tpr - 1) ** 2 + fpr**2)]
        else:
            threshold = (y_score[y_true].min() + y_score[~y_true].max()) / 2
        y_pred = y_score > threshold
        self.auc = auc
        self.threshold = threshold
        self.accuracy_score = accuracy_score(y_true, y_pred)
        self.confusion_matrix = pd.Series(
            data=confusion_matrix(y_true, y_pred).ravel(),
            index=["true_neg", "false_pos", "false_neg", "true_pos"],
        )

    def accuracy_confusion(self, data, labels):
        y_true = labels
        y_pred = self.predict(data)
        accuracy_score = accuracy_score(y_true, y_pred)
        confusion_matrix = pd.Series(
            data=confusion_matrix(y_true, y_pred).ravel(),
            index=["true_neg", "false_pos", "false_neg", "true_pos"],
        )
        return accuracy_score, confusion_matrix

    def score(self, data):
        if any(i not in data.columns for i in self.feature_names_in_):
            raise ValueError("input data missing features")
        ImAgeDistance = data[self.feature_names_in_].apply(
            self._scalar_projection, axis=1
        )
        ImAgeDistance.name = "ImAge"
        return ImAgeDistance

    def score_orthogonal(self, data):
        if any(i not in data.columns for i in self.feature_names_in_):
            raise ValueError("input data missing features")
        ImAgeOrthogonal = data[self.feature_names_in_].apply(
            self._ortho_distance, axis=1
        )
        ImAgeOrthogonal.name = "ImAgeOrthogonal"
        return ImAgeOrthogonal

    def project_orthogonal_subspace(self, data):
        if any(i not in data.columns for i in self.feature_names_in_):
            raise ValueError("input data missing features")
        ImAgeOrthogonal = data.apply(self._ortho_projection, axis=1)
        # ImAgeOrthogonal.name = "ImAgeOrthogonal"
        return ImAgeOrthogonal

    def predict(self, data):
        y_pred = self.score(data) > self.threshold
        y_pred.name = f"prob_{self.classes_[1]}"
        return y_pred

    def fit_score(
        self,
        data: pd.DataFrame,
        group_by: list,
        group_A: tuple,
        group_B: tuple,
    ):
        self.fit(data, group_by, group_A, group_B)
        return pd.concat([self.scores, ImAgeOrthogonal], axis=1, join="outer")
