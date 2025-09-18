
from fastapi import FastAPI
from pydantic import BaseModel, EmailStr
import xgboost as xgb
import numpy as np
import torch
from model_train import LSTMModel
import os

app = FastAPI(title="FluPredict API Prototype")

# Load models if present, otherwise advise running model_train.py
xgb_model = None
lstm = None
if os.path.exists('models/xgb.json'):
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model('models/xgb.json')
if os.path.exists('models/lstm.pth'):
    lstm = LSTMModel(input_dim=3)
    lstm.load_state_dict(torch.load('models/lstm.pth'))
    lstm.eval()

class RegisterRequest(BaseModel):
    first_name: str
    email: EmailStr

class PredictRequest(BaseModel):
    lag1: float
    lag2: float
    weekofyear: int

@app.post("/register")
def register(req: RegisterRequest):
    # For prototype, we simply echo
    return {"status":"ok","user":req.email, "first_name": req.first_name}

@app.post("/predict")
def predict(req: PredictRequest):
    if xgb_model is None or lstm is None:
        return {"error":"Models not found. Run `python model_train.py` in backend/ to create placeholder models."}
    x = np.array([[req.lag1, req.lag2, req.weekofyear]])
    xgb_pred = xgb_model.predict(x)[0]
    seq = np.tile(x, (1,4,1))
    with torch.no_grad():
        lstm_pred = lstm(torch.tensor(seq, dtype=torch.float32)).numpy().flatten()[0]
    ensemble = 0.6 * lstm_pred + 0.4 * xgb_pred
    return {"xgb": float(xgb_pred), "lstm": float(lstm_pred), "ensemble": float(ensemble)}
