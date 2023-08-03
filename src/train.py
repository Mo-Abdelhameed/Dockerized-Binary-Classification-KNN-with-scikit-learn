import os
from KNN_Classifier import Classifier
from utils import read_csv_in_directory
from config import paths
from logger import get_logger, log_error
from schema.data_schema import load_json_data_schema, save_schema
from preprocessing.pipeline import create_pipeline
from preprocessing.preprocess import percentage_of_missing_values

logger = get_logger(task_name="train")


def run_training():
    try:
        logger.info("Starting training...")

        logger.info("Loading and saving schema...")
        data_schema = load_json_data_schema(paths.INPUT_SCHEMA_DIR)
        save_schema(schema=data_schema, save_dir_path=paths.SAVED_SCHEMA_DIR_PATH)

        logger.info("Loading training data...")
        train_data = read_csv_in_directory(paths.TRAIN_DIR)
        features = data_schema.features
        target = data_schema.target
        x_train = train_data[features]
        y_train = train_data[target]
        pipeline = create_pipeline(data_schema)
        for stage, column in pipeline:
            if column is None:
                x_train = stage(x_train)
            elif column == 'schema':
                x_train = stage(x_train, data_schema)
            else:
                if stage.__name__ == 'remove_outliers_zscore':
                    x_train, y_train = stage(x_train, column, target=y_train)
                else:
                    x_train = stage(x_train, column)
        model = Classifier()
        model.fit(x_train, y_train)
        if not os.path.exists(paths.PREDICTOR_DIR_PATH):
            os.makedirs(paths.PREDICTOR_DIR_PATH)
        model.save(paths.PREDICTOR_DIR_PATH)
        logger.info('Model saved!')

    except Exception as exc:
        err_msg = "Error occurred during training."
        # Log the error
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file
        log_error(message=err_msg, error=exc, error_fpath=paths.TRAIN_ERROR_FILE_PATH)
        # re-raise the error
        raise Exception(f"{err_msg} Error: {str(exc)}") from exc


if __name__ == "__main__":
    run_training()
