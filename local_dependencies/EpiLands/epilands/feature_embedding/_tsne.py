from __future__ import annotations

# Import libraries
import pandas as pd
from sklearn.manifold import TSNE

# Import libraries
import pandas as pd

# Relative imports
# from ..get import get_feature_cols
from ..config import NAME_SEPARATOR_
from ..generic_read_write import save_dataframe_to_csv


def generate_tsne(
    df_input: pd.DataFrame,
    save_to: str,
    *,
    feature_cols: list = None,
    perplexity="default",
    metric="euclidean",
    n_components: int = 2,
    group_col_name: str = "Group",
    save_fit_data: bool = True,
    **kwargs,
):
    # if feature_cols == None:
    #     print(
    #         "NOTICE: No feature cols provided, using the default function MIEL_general.get_feature_cols(df)"
    #     )
    #     feature_cols = get_feature_cols(df_input)
    if perplexity == "default":
        perplexity = int(len(df_input.index) * 0.05)

    tsne = TSNE(
        n_components=n_components, perplexity=perplexity, metric=metric, **kwargs
    )
    tsne_fit = tsne.fit_transform(df_input)

    df_tsne = pd.DataFrame(index=df_input.index)
    df_tsne[["tSNE{}".format(i + 1) for i in range(n_components)]] = tsne_fit
    df_tsne.sort_values(by=group_col_name, inplace=True)
    df_tsne.attrs["name"] = NAME_SEPARATOR_.join((df_input.attrs["name"], "tSNE"))

    if save_fit_data:
        save_dataframe_to_csv(df_tsne, save_to)

    return tsne, df_tsne
