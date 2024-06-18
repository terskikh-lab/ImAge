# Import libraries
from __future__ import annotations
from typing import Tuple, Union
import pandas as pd
from sklearn.decomposition import FastICA, PCA

# Relative imports


def calculate_component_analysis(
    fit_data: pd.DataFrame,
    feature_data: pd.DataFrame,
    analysis_type: str,
    n_components: int,
) -> Tuple[Union[PCA, FastICA], pd.DataFrame]:
    """
    Calculates the component analysis for the given analysis type and adds it to the anndata object.
    \nAdds the component analysis fit to obsm and adds the parameter data to uns.

    adata: anndata object

    analysis_type: str one of ['PCA', 'ICA']

    n_components: int number of components to reduce to

    save_info: bool if True the fit and parameter data will be saved to uns

    save_to: str path to the folder where the graphs will be saved or location in adata.uns
    """
    if analysis_type == "ICA":
        component_analysis = FastICA(n_components=n_components).fit(fit_data.values)
        fit = component_analysis.transform(feature_data)
        uns_data = {
            "params": {"components": n_components},
            "whitening_": component_analysis.whitening_,
            "labels": [
                analysis_type.replace("A", "") + str(comp_i + 1)
                for comp_i in range(n_components)
            ],
        }
    if analysis_type == "PCA":
        component_analysis = PCA(n_components=n_components).fit(fit_data.values)
        fit = component_analysis.transform(feature_data)
        uns_data = {
            "params": {"n_components": n_components},
            "variance": component_analysis.explained_variance_,
            "variance_ratio": component_analysis.explained_variance_ratio_,
            "labels": [
                analysis_type.replace("A", "") + str(i + 1) + f"({var*100:.1f}%)"
                for i, var in enumerate(component_analysis.explained_variance_ratio_)
            ],
        }
    return fit, uns_data, component_analysis
