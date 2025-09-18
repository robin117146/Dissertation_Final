
# Simple training script that creates placeholder models for the prototype.
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
import os

os.makedirs('models', exist_ok=True)

def load_data():
    rng = pd.date_range('2020-01-01', periods=300, freq='W')
    cases = (np.sin(np.linspace(0,20,300)) * 200 + 1500 + np.random.randn(300)*50).astype(int)
    df = pd.DataFrame({"week": rng, "cases": cases})
    df['lag1'] = df['cases'].shift(1).fillna(method='bfill')
    df['lag2'] = df['cases'].shift(2).fillna(method='bfill')
    df['weekofyear'] = df['week'].dt.isocalendar().week
    return df

class LSTMModel(nn.Module):
    def __init__(self,input_dim,hidden=32):
        super().__init__()
        self.lstm=nn.LSTM(input_dim,hidden, batch_first=True)
        self.fc=nn.Linear(hidden,1)
    def forward(self,x):
        out,_ = self.lstm(x)
        return self.fc(out[:,-1,:])

def train_xgb(df):
    X = df[['lag1','lag2','weekofyear']].values
    y = df['cases'].values
    split = int(len(X)*0.8)
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]
    model = xgb.XGBRegressor(n_estimators=50, max_depth=4)
    model.fit(Xtr, ytr, eval_set=[(Xte,yte)], early_stopping_rounds=5, verbose=False)
    preds = model.predict(Xte)
    print("XGB MAE:", mean_absolute_error(yte, preds))
    model.save_model('models/xgb.json')
    return model

def train_lstm(df):
    values = df[['cases','lag1','lag2']].values.astype(float)
    seq_len=4
    X=[]; y=[]
    for i in range(len(values)-seq_len):
        X.append(values[i:i+seq_len])
        y.append(values[i+seq_len,0])
    X=np.array(X); y=np.array(y)
    split=int(len(X)*0.8)
    Xtr,Xte = X[:split],X[split:]
    ytr,yte = y[:split],y[split:]
    model=LSTMModel(input_dim=3)
    opt=torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn=nn.MSELoss()
    for epoch in range(50):
        model.train()
        xb=torch.tensor(Xtr, dtype=torch.float32)
        yb=torch.tensor(ytr, dtype=torch.float32).unsqueeze(-1)
        pred=model(xb)
        loss=loss_fn(pred,yb)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(Xte, dtype=torch.float32)).numpy().flatten()
    print("LSTM MAE:", mean_absolute_error(yte, preds))
    torch.save(model.state_dict(), 'models/lstm.pth')
    return model

if __name__=='__main__':
    df = load_data()
    train_xgb(df)
    train_lstm(df)
    print('Models saved to models/ (xgb.json, lstm.pth)')
