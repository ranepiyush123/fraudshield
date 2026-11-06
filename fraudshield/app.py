import sqlite3
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="FraudShield", version="1.0.0")
DB = Path("transactions.db")

class Transaction(BaseModel):
    amount: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    international: bool = False
    new_device: bool = False
    previous_transactions: int = Field(ge=0)

def features(t: Transaction):
    return [
        t.amount,
        t.hour,
        int(t.international),
        int(t.new_device),
        t.previous_transactions,
    ]

def train_demo_model():
    rng = np.random.default_rng(42)
    n = 3000
    amount = rng.exponential(120, n)
    hour = rng.integers(0, 24, n)
    international = rng.integers(0, 2, n)
    new_device = rng.integers(0, 2, n)
    previous = rng.integers(0, 100, n)

    X = np.column_stack(
        [amount, hour, international, new_device, previous]
    )
    risk = (
        (amount > 400).astype(int) * 2
        + international
        + new_device
        + ((hour < 5) | (hour > 22)).astype(int)
        + (previous < 3).astype(int)
    )
    y = (risk >= 3).astype(int)
    model = RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    model.fit(X, y)
    return model

model = train_demo_model()

def database():
    connection = sqlite3.connect(DB)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            amount REAL,
            probability REAL,
            risk TEXT
        )
    """)
    connection.commit()
    return connection

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(transaction: Transaction):
    probability = float(
        model.predict_proba([features(transaction)])[0][1]
    )
    risk = (
        "High" if probability >= 0.7
        else "Medium" if probability >= 0.3
        else "Low"
    )
    with database() as conn:
        conn.execute(
            "INSERT INTO transactions (amount, probability, risk) "
            "VALUES (?, ?, ?)",
            (transaction.amount, probability, risk),
        )
    return {
        "fraud_probability": round(probability, 4),
        "risk_level": risk,
        "model_type": "synthetic_demo",
    }

@app.get("/transactions")
def transactions():
    with database() as conn:
        rows = conn.execute(
            "SELECT id, amount, probability, risk "
            "FROM transactions ORDER BY id DESC LIMIT 50"
        ).fetchall()
    return [
        {
            "id": r[0],
            "amount": r[1],
            "probability": r[2],
            "risk": r[3],
        }
        for r in rows
    ]

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return Path("fraudshield/dashboard.html").read_text()
