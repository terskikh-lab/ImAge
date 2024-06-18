# Import libraries
from __future__ import annotations
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.manifold import MDS

# Relative imports
from ..generic_read_write import save_dataframe_to_csv


def generate_mds(
    df_pdist: pd.DataFrame,
    metric: bool = True,
    n_components: int = 2,
    save_to: str = None,
    save_info: bool = True,
    **kwargs,
) -> Tuple[MDS, pd.DataFrame, pd.DataFrame]:
    """
    Function to calculate the multidimensional scaling of a distance matrix.
    The function creates a 2-dimensional representation of the data based on the
    distance matrix.

    Parameters:
    df_pdist (DataFrame): The distance matrix between all of the data points.
    The rows and columns of the matrix should be the same.

    Returns:
    df_mds (DataFrame): A DataFrame containing the two-dimensional
    representation of the data.
    df_mds_params (DataFrame): A DataFrame containing the stress and stress1 (Kruskals)
    of the data, as well as the number of iterations to find the optimal stress.
    """
    mds = MDS(
        n_components=n_components, metric=metric, dissimilarity="precomputed", verbose=1
    )
    df_mds = pd.DataFrame(index=df_pdist.index)
    mds_fit = mds.fit_transform(df_pdist.values)
    stress = mds.stress_
    df_mds[["MDS{}".format(i + 1) for i in range(n_components)]] = mds_fit
    df_mds.attrs["name"] = df_pdist.attrs["name"] + "_MDS"
    # have to do manually since sklearn outputs raw stress
    # found here: http://eric.univ-lyon2.fr/~ricco/tanagra/fichiers/fr_Tanagra_Community_Detection_MDS.pdf
    stress1 = np.sqrt(stress / (0.5 * np.sum(df_pdist.values**2)))
    if stress1 > 0.15:
        print(
            "Warning: STRESS1 (Kruskals) is greater than 0.15. Remember, stress1 < 0.1 is excellent, stress1 < 0.15 is acceptable, and stress1 > 0.15 is the threshold for distance preservation"
        )
    df_mds.attrs["name"] = df_pdist.attrs["name"] + "_MDS"
    df_mds_params = pd.DataFrame.from_dict(
        {"Stress": [stress], "Kruskal's Stress": [stress1], "Iterations": [mds.n_iter_]}
    )
    df_mds_params.attrs["name"] = df_mds.attrs["name"] + "_params"
    if save_info:
        save_dataframe_to_csv(df_mds_params, save_to)
        save_dataframe_to_csv(df_mds, save_to)
    print("MDS parameters: ", df_mds_params)
    # display(df_mds)
    return mds, df_mds, df_mds_params
