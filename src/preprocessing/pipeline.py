import pandas as pd
from typing import List, Any
from src.schema.data_schema import BinaryClassificationSchema


def create_pipeline(input_data: pd.DataFrame, input_schema: BinaryClassificationSchema) -> List[Any]:
    """
        Creates pipeline of preprocessing steps

        Args:
            input_data (pd.Dataframe): Dataframe of input data.
            input_schema (BinaryClassificationSchema): BinaryClassificationSchema object carrying data about the schema
        Returns:
            A list of tuples containing the functions to be executed in the pipeline on a certain column
        """

    pass
