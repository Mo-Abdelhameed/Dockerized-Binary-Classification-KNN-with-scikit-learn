import numpy as np
import pandas as pd
from typing import List
from config import paths
from utils import read_csv_in_directory, read_json_as_dict, save_dataframe_as_csv
from logger import get_logger
from KNN_Classifier import Classifier
from preprocessing.pipeline import create_pipeline
from schema.data_schema import load_json_data_schema, load_saved_schema
from joblib import load
from preprocessing.preprocess import normalize

logger = get_logger(task_name="predict")


def create_predictions_dataframe(
        predictions_arr: np.ndarray,
        class_names: List[str],
        prediction_field_name: str,
        ids: pd.Series,
        id_field_name: str,
        return_probs: bool = False,
) -> pd.DataFrame:
    """
    Converts the predictions numpy array into a dataframe having the required structure.

    Performs the following transformations:
    - converts to pandas dataframe
    - adds class labels as headers for columns containing predicted probabilities
    - inserts the id column

    Args:
        predictions_arr (np.ndarray): Predicted probabilities from predictor model.
        class_names List[str]: List of target classes (labels).
        prediction_field_name (str): Field name to use for predicted class.
        ids: ids as a numpy array for each of the samples in  predictions.
        id_field_name (str): Name to use for the id field.
        return_probs (bool, optional): If True, returns the predicted probabilities
            for each class. If False, returns the final predicted class for each
            data point. Defaults to False.

    Returns:
        Predictions as a pandas dataframe
    """
    if predictions_arr.shape[1] != len(class_names):
        raise ValueError(
            "Length of class names does not match number of prediction columns"
        )
    predictions_df = pd.DataFrame(predictions_arr, columns=class_names)
    if len(predictions_arr) != len(ids):
        raise ValueError("Length of ids does not match number of predictions")
    predictions_df.insert(0, id_field_name, ids)
    if return_probs:
        return predictions_df
    predictions_df[prediction_field_name] = predictions_df[class_names].idxmax(axis=1)
    predictions_df.drop(class_names, axis=1, inplace=True)
    return predictions_df


def run_batch_predictions() -> None:
    test_data = read_csv_in_directory(paths.TEST_DIR)
    data_schema = load_saved_schema(paths.SAVED_SCHEMA_DIR_PATH)
    model = Classifier.load(paths.PREDICTOR_DIR_PATH)
    pipeline = create_pipeline(test_data, data_schema)
    features = data_schema.features
    x_test = test_data[features]
    logger.info("Transforming the data...")
    for stage, column in pipeline:
        if column is None:
                x_test = stage(x_test)
        elif column == 'schema':
            if stage.__name__ == 'normalize':
                try:
                    scaler = load(paths.SCALER_FILE)
                    x_test = normalize(x_test, data_schema, scaler)
                except:
                     pass
            elif stage.__name__ == 'encode':
                x_test = stage(x_test, data_schema, encoder='predict')
            else:
                x_test = stage(x_test, data_schema)
        else:
            if stage.__name__ == 'remove_outliers_zscore':
                    x_test, _ = stage(x_test, column)
            else:
                    x_test = stage(x_test, column)

    logger.info("Making predictions...")
    predictions_arr = Classifier.predict_with_model(model, x_test, return_probs=True)
    predictions_df = create_predictions_dataframe(
        predictions_arr,
        data_schema.target_classes,
        'prediction',
        test_data[data_schema.id],
        data_schema.id,
        return_probs=True,
    )

    logger.info("Saving predictions...")
    save_dataframe_as_csv(
        dataframe=predictions_df, file_path=paths.PREDICTIONS_FILE_PATH
    )

    logger.info("Batch predictions completed successfully")


run_batch_predictions()
