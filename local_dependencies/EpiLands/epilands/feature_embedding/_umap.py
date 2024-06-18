from __future__ import annotations

# Import libraries
import pandas as pd
from umap import UMAP

# Import libraries
import pandas as pd

# Relative imports
# from ..get import get_feature_cols
from ..config import NAME_SEPARATOR_
from ..generic_read_write import save_dataframe_to_csv


def generate_umap(
    df: pd.DataFrame,
    save_to: str,
    *,
    feature_cols: list = None,
    n_neighbors: int = 15,
    n_components: int = 2,
    metric: str = "euclidean",
    output_metric: str = "euclidean",
    min_dist: float = 0.1,
    spread: float = 1.0,
    save_fit_data: bool = True,
    **kwargs,
):
    Umap = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric=metric,
        output_metric=output_metric,
        min_dist=min_dist,
        spread=spread,
    )
    # if feature_cols is None:
    #     print(
    #         "NOTICE: No feature cols provided, using the default function get.get_feature_cols(df)"
    #     )
    #     feature_cols = get_feature_cols(df)
    umap_fit = Umap.fit(df[feature_cols])

    df_umap = pd.DataFrame(index=df.index)
    df_umap[["UMAP{}".format(i + 1) for i in range(n_components)]] = umap_fit.embedding_
    df_umap.attrs["name"] = NAME_SEPARATOR_.join((df.attrs["name"], "UMAP"))

    if save_fit_data:
        save_dataframe_to_csv(df_umap, save_to)
    return Umap, df_umap
