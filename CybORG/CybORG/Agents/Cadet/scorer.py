import argparse
from typing import Dict

import pandas as pd
from CybORG.Agents.Cadet.impact_reporters import (AVAIL_DROP_KEY, CONF_DROP_KEY, IMPACT_TYPE_KEY,
                              INTEGRITY_DROP_KEY)
from CybORG.Agents.Cadet.scorer_mappings import (AVAIL_MAP_ORIGINAL, AVAIL_MAP_V2, AVAIL_MAP_V3,
                             AVAIL_MAP_V4, CONF_MAP_ORIGINAL, CONF_MAP_V2,
                             CONF_MAP_V3, INTEG_MAP_ORIGINAL, INTEG_MAP_V2,
                             INTEG_MAP_V3)


def verify_then_apply_mapping(
    df: pd.DataFrame, column_name: str, mapping: dict
) -> pd.DataFrame:
    """Verify that the mapping covers all the values in the column, and then apply the mapping.

    Args:
        df (pd.DataFrame): The dataframe to apply the mapping to.
        column_name (str): The column to apply the mapping to.
        mapping (dict): The mapping to apply.

    Raises:
        ValueError: If the mapping does not cover all values in the column.

    Returns:
        pd.DataFrame: The dataframe with the mapping applied.
    """
    # We need to check that the mapping cover all the values
    unique_values = set(df[column_name].unique())
    mapping_values = set(mapping.keys())
    # The unique values should be a subset of the mapping values
    if not unique_values.issubset(mapping_values):
        raise ValueError(
            f"Mapping does not cover all values: {unique_values - mapping_values} for column {column_name}"
        )
    # Apply the mapping
    df[column_name] = df[column_name].map(mapping)
    return df


class Scorer:
    def __init__(self, mapping: Dict[int, float], in_column: str, out_column: str):
        self.mapping = mapping
        self.in_column = in_column
        self.out_column = out_column

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.out_column] = df[self.in_column].map(self.mapping)
        return df

    def __repr__(self):
        return f"{self.__class__.__name__}(mapping={self.mapping}, in_column={self.in_column}, out_column={self.out_column})"

    @property
    def max_value(self):
        return max(self.mapping.values())


class AvailabilityScorer(Scorer):
    def __init__(self, mapping):
        super().__init__(mapping, IMPACT_TYPE_KEY, AVAIL_DROP_KEY)


class IntegrityScorer(Scorer):
    def __init__(self, mapping):
        super().__init__(mapping, IMPACT_TYPE_KEY, INTEGRITY_DROP_KEY)


class ConfidentialityScorer(Scorer):
    def __init__(self, mapping):
        super().__init__(mapping, IMPACT_TYPE_KEY, CONF_DROP_KEY)


class ScorerChain:
    def __init__(self, scorers: list[Scorer]):
        self.scorers = scorers

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        for scorer in self.scorers:
            df = scorer(df)
        return df

    def __repr__(self):
        return f"{self.__class__.__name__}(scorers={self.scorers})"

    @property
    def max_values(self) -> dict[str, float]:
        return {scorer.out_column: scorer.max_value for scorer in self.scorers}


ORIGINAL_SCORER = ScorerChain(
    [
        AvailabilityScorer(AVAIL_MAP_ORIGINAL),
        IntegrityScorer(INTEG_MAP_ORIGINAL),
        ConfidentialityScorer(CONF_MAP_ORIGINAL),
    ]
)

SCORER_V2 = ScorerChain(
    [
        AvailabilityScorer(AVAIL_MAP_V2),
        IntegrityScorer(INTEG_MAP_V2),
        ConfidentialityScorer(CONF_MAP_V2),
    ]
)

SCORER_V3 = ScorerChain(
    [
        AvailabilityScorer(AVAIL_MAP_V3),
        IntegrityScorer(INTEG_MAP_V3),
        ConfidentialityScorer(CONF_MAP_V3),
    ]
)

SCORER_V4 = ScorerChain(
    [
        AvailabilityScorer(AVAIL_MAP_V4),
        IntegrityScorer(INTEG_MAP_V3),
        ConfidentialityScorer(CONF_MAP_V3),
    ]
)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file")
    parser.add_argument("output_csv_path", type=str, help="Path to the output CSV file")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    df = pd.read_csv(args.csv_path)

    scorers = ORIGINAL_SCORER

    for scorer in scorers:
        df = scorer(df)
    df.to_csv(args.output_csv_path, index=False)
