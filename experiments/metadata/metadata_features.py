import numpy as np
import pandas as pd
import torch


SEX_CATEGORIES = [
    "female",
    "male",
    "unknown",
]

LOCALIZATION_CATEGORIES = [
    "abdomen",
    "acral",
    "back",
    "chest",
    "ear",
    "face",
    "foot",
    "genital",
    "hand",
    "lower extremity",
    "neck",
    "scalp",
    "trunk",
    "unknown",
    "upper extremity",
]


class MetadataFeatureEncoder:
    def __init__(self):
        self.age_median = None
        self.age_mean = None
        self.age_std = None
        self.is_fitted = False

    def fit(self, metadata):
        ages = pd.to_numeric(
            metadata["age"],
            errors="coerce"
        )

        self.age_median = ages.median()

        filled_ages = ages.fillna(
            self.age_median
        )

        self.age_mean = filled_ages.mean()
        self.age_std = filled_ages.std()

        if self.age_std == 0:
            self.age_std = 1.0

        self.is_fitted = True

        return self

    def transform(self, metadata):
        if not self.is_fitted:
            raise RuntimeError(
                "MetadataFeatureEncoder must be fitted before transform()."
            )

        ages = pd.to_numeric(
            metadata["age"],
            errors="coerce"
        )

        age_missing = ages.isna().astype(np.float32)

        filled_ages = ages.fillna(
            self.age_median
        )

        normalized_age = (
            filled_ages - self.age_mean
        ) / self.age_std

        sex_values = (
            metadata["sex"]
            .fillna("unknown")
            .replace("", "unknown")
        )

        localization_values = (
            metadata["localization"]
            .fillna("unknown")
            .replace("", "unknown")
        )

        sex_features = np.stack([
            (sex_values == category).astype(np.float32)
            for category in SEX_CATEGORIES
        ], axis=1)

        localization_features = np.stack([
            (localization_values == category).astype(np.float32)
            for category in LOCALIZATION_CATEGORIES
        ], axis=1)

        features = np.column_stack([
            normalized_age.to_numpy(dtype=np.float32),
            age_missing.to_numpy(dtype=np.float32),
            sex_features,
            localization_features,
        ])

        return torch.tensor(
            features,
            dtype=torch.float32
        )