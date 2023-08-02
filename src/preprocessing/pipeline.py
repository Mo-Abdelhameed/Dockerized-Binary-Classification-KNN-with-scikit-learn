from typing import List
from schema.data_schema import BinaryClassificationSchema
from preprocessing.preprocess import *
from scipy.stats import shapiro


def create_pipeline(input_data: pd.DataFrame, schema: BinaryClassificationSchema) -> List[Any]:
    """
        Creates pipeline of preprocessing steps

        Args:
            input_data (pd.Dataframe): Dataframe of input data.
            schema (BinaryClassificationSchema): BinaryClassificationSchema object carrying data about the schema
        Returns:
            A list of tuples containing the functions to be executed in the pipeline on a certain column
        """
    pipeline = [(drop_constant_features, None),
                (drop_all_nan_features, None),
                (drop_duplicate_features, None),
                (drop_mostly_missing_columns, None),
                (indicate_missing_values, None),
                ]
    numeric_features = schema.numeric_features
    cat_features = schema.categorical_features
    for f in numeric_features:
        pipeline.append((impute_numeric, f))
        pipeline.append((remove_outliers_zscore, f))
    pipeline.append((normalize, 'schema'))

    for f in cat_features:
        pipeline.append((impute_categorical, f))
    pipeline.append((encode, 'schema'))

    return pipeline
