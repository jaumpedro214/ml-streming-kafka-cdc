import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from collections import defaultdict

import bentoml

from datetime import datetime
import os


class DataframeFrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Class to encode categorical variables into frequency values.
        """
        pass

    def fit(self, X, y=None):
        self.columns = X.columns
        self.mapping = {
            column: X[column].value_counts(normalize=True)
            for column in self.columns
        }
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        for column in self.columns:
            X_[column] = X_[column].map(
                self.mapping[column]
            )
        return X_


class RemoveUncommonClasses(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[], frequencies=[]):
        """
        Class to remove uncommon categories from a list of columns

        Parameters
        ----------
        columns : list of str, optional
            list of columns to remove uncommon classes from, by default []
        frequencies : list of floats or list of ints, optional
            the minimum frequency of the category to be kept, by default []
        """
        self.columns = columns
        self.frequencies = frequencies
        self.replace_rules = dict()

    def get_params(self, deep=True):
        return {"columns": self.columns,
                "frequencies": self.frequencies
                }

    def fit(self, df, y=None):
        df_ = df.copy()
        for column, frequency in zip(self.columns, self.frequencies):
            self.build_replace_dict(df_, column, frequency)
        return self
    
    def return_other(self):
        return "Other"

    def build_replace_dict(self, df, column, frequency):
        """
        Build a dictionary of replacements for a column

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe to build the replace dict from
        column : str
            column to build the replace dict from
        frequency : float or int
            minimum frequency of the category to be kept
        """

        # If the frequency is a float, then normalize the counts to a percentage
        replace_dict = df[column].value_counts(
            normalize=isinstance(frequency, float)
        )\
            .to_dict()

        replace_dict = defaultdict(
            self.return_other, 
            replace_dict
        )

        # If the categoy frequency is higher than the threshold, then keep it
        # otherwise, replace it with the replace_by token
        for category, count in replace_dict.items():
            replace_dict[category] = category if count > frequency else self.return_other()

        self.replace_rules[column] = replace_dict

    def transform(self, df, y=None):
        df_ = df.copy()
        return df_.replace(self.replace_rules)


if __name__ == "__main__":
    # read test data from csv file ford_test.csv
    PATH = os.path.join(os.path.dirname(__file__), "data/ford_test.csv")
    CATEGORICAL_VARS = ['model', 'transmission', 'fuelType']

    ford_df = pd.read_csv(PATH)
    
    # Define the transformer pipeline for the dataframe
    frequency_encoder = DataframeFrequencyEncoder()
    remove_uncommon_classes = RemoveUncommonClasses(
        CATEGORICAL_VARS, [0.01, 0, 20]
    )
    feature_eng_tfm = ColumnTransformer(
        [
            (
                "Mileage",
                Pipeline([
                    ("Log1P", FunctionTransformer(np.log1p)),
                    ("Scaler", StandardScaler())
                ]),
                ["mileage"]
            ),
            (
                "CatFeatures",
                Pipeline([
                    ("DataCleaning", remove_uncommon_classes),
                    ("FrequencyEncoder", frequency_encoder),
                ]),
                CATEGORICAL_VARS
            ),
            (
                "Year",
                Pipeline([
                    ("ClipValues", MinMaxScaler(
                        feature_range=(2000, 2020), clip=True)),
                    ("Scaler", StandardScaler())
                ]),
                ["year"]
            ),
        ],
        remainder=StandardScaler()
    )
    
    # Define the regressor model
    xgb_reg = XGBRegressor(
        max_depth=None, 
        n_estimators=100,
        random_state=214,
    )
    
    ml_model = Pipeline([
        ("FeatureEngineering", feature_eng_tfm),
        ("Regressor", xgb_reg)
    ])
    
    # Train the model to predict the price of the car
    print(f"[{datetime.now()}] Training the model...")
    
    ml_model.fit(
        ford_df.drop(columns=["price"]), 
        ford_df["price"]
    )
    
    print(f"[{datetime.now()}] Saving the model...")
    
    # Save the model to a bento service
    bentoml.sklearn.save_model(
        "ford_price_predictor", 
        ml_model,
        custom_objects={
            "DataframeFrequencyEncoder": frequency_encoder,
            "RemoveUncommonClasses": remove_uncommon_classes,
        }
    )
