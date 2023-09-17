import uvicorn
from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
from pydantic import BaseModel

description = """
Welcome to the GetAround API.
"""

tags_metadata = [
    {
        "name": "Default",
        "description": "Default endpoint"
    },

    {
        "name": "Machine-Learning",
        "description": "Endpoints that deal with price prediction",
    }
]



app = FastAPI(
    title="GetAround",
    description=description,
    openapi_tags=tags_metadata
)

@app.get("/", tags=["Default"])
async def index():

    message = "Hello world! This `/` is the most simple and default endpoint. If you want to learn more, check out documentation of the api at `/docs`"

    return message

class predictionFeatures(BaseModel):
    model_key: str
    mileage: int
    engine_power: int
    fuel: str
    paint_color: str
    car_type: str
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool

@app.post("/predict", tags=["Machine-Learning"])
async def predict(input_pred: predictionFeatures):

    df = pd.DataFrame(dict(input_pred), index=[0])

    logged_model = 'mlruns/515851487349325251/33a6953cd77c4f52887edc4d5838b7ab/artifacts/modeling_car_price'

    loaded_model = mlflow.pyfunc.load_model(logged_model)
    prediction = loaded_model.predict(df)
    
    response = {"prediction": prediction.tolist()[0]}
    return response

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)