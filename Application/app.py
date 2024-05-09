"""
@Author: Canmert Demir
@Date: 2024-05-02
@Email: canmertdemir2@gmail.com
# pip install fastapi uvicorn python-multipart => Fast Api Install

To run app : uvicorn app:app --reload
"""


from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import torch

app = FastAPI()

# Assuming your model is loaded from a file named 'linear_regression_model.pth'
model = torch.load('../Deep_Learning_Model/linear_regression_model.pth')

def variable_fix(input_data):
    # Eğer input_data üzerinde herhangi bir işlem yapılacaksa, burada yapılabilir
    return input_data

@app.post("/predict/")
async def predict_image(
    Date: str = Form(...),
    Time: str = Form(...),
    Global_active_power: float = Form(...),
    Global_reactive_power: float = Form(...),
    Voltage: float = Form(...),
    Global_intensity: float = Form(...),
    Sub_metering_1: float = Form(...),
    Sub_metering_2: float = Form(...),
    Sub_metering_3: float = Form(...),
):
    # Prepare input data
    input_data = pd.DataFrame({
        'Date': [Date],
        'Time': [Time],
        'Global_active_power': [Global_active_power],
        'Global_reactive_power': [Global_reactive_power],
        'Voltage': [Voltage],
        'Global_intensity': [Global_intensity],
        'Sub_metering_1': [Sub_metering_1],
        'Sub_metering_2': [Sub_metering_2],
        'Sub_metering_3': [Sub_metering_3],
    })
    input_data = variable_fix(input_data)
    input_tensor = torch.tensor(input_data.values).float()

    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor)

    # Return prediction
    return JSONResponse(content={"Date": Date, "Time": Time, "Prediction": prediction.item()})



