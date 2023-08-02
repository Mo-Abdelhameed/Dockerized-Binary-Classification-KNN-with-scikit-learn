import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from config import paths
from logger import get_logger, log_error

logger = get_logger(task_name='serve')

def create_app(model_resources):

    app = FastAPI()

    @app.get('/ping')
    async def ping() -> dict:
        """GET endpoint that returns a message indicating the service is running.

        Returns:
            dict: A dictionary with a "message" key and "Pong!" value.
        """
        logger.info("Received ping request. Service is healthy...")
        return {"message": "Pong!"}
