import numpy as np
import pandas as pd

from CybORG.Agents.Cadet.scorer import AVAIL_DROP_KEY, CONF_DROP_KEY, INTEGRITY_DROP_KEY


def weighted_sum(df: pd.DataFrame, name_to_weight: dict[str, float]) -> pd.DataFrame:
    """Given a list of pairs (name, weight), where the name are the columns of the dataframe, I want to get the weighted sum of the dataframe.

    Args:
        df (pd.DataFrame): The dataframe to weight.
        name_to_weight (dict[str, float]): The mapping from the column name to the weight.

    Returns:
        pd.DataFrame: The dataframe with the weighted sum.
    """
    weighted = df[["steps"]]
    weighted["weighted"] = np.zeros(len(df))
    for column, factor in name_to_weight.items():
        weighted["weighted"] += df[column].values * factor
    return weighted


class Weighter:
    """A class to weight the scores of the different resilience sub metrics."""

    def __init__(self, name_to_weight: dict[str, float]):
        self.name_to_weight = name_to_weight

    def __call__(self, df: pd.DataFrame):
        assert set(self.name_to_weight.keys()).issubset(df.columns)
        assert "steps" in df.columns
        return weighted_sum(df, self.name_to_weight)

    def compute_max_value(self, name_to_max_value: dict[str, float]):
        assert set(self.name_to_weight.keys()) == set(name_to_max_value.keys())
        return sum(
            self.name_to_weight[column] * name_to_max_value[column]
            for column in self.name_to_weight
        )


ORIGINAL_WEIGHTER = Weighter(
    {
        CONF_DROP_KEY: 1 / 3,
        INTEGRITY_DROP_KEY: 1 / 3,
        AVAIL_DROP_KEY: 1 / 3,
    }
)

WEIGHTER_V2 = Weighter(
    {
        CONF_DROP_KEY: 0.6,
        INTEGRITY_DROP_KEY: 0.2,
        AVAIL_DROP_KEY: 0.2,
    }
)

WEIGHTER_V3 = Weighter(
    {
        CONF_DROP_KEY: 0.1,
        INTEGRITY_DROP_KEY: 0.1,
        AVAIL_DROP_KEY: 0.8,
    }
)

WEIGHTER_V4 = Weighter(
    {
        CONF_DROP_KEY: 0,
        INTEGRITY_DROP_KEY: 0,
        AVAIL_DROP_KEY: 1,
    }
)
