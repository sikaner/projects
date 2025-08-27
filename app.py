import os
from typing import List


from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import mlflow.pyfunc as pyfunc
from openai import OpenAI


# Load .env values (e.g., OPENAI_API_KEY, MODEL_DIR)
load_dotenv()


APP_PORT = int(os.getenv("APP_PORT", "8000"))
MODEL_DIR = os.getenv("MODEL_DIR", "./model")


# Load MLflow model once at startup
model = pyfunc.load_model(MODEL_DIR)


# OpenAI client (reads OPENAI_API_KEY from environment)
client = OpenAI()


app = FastAPI(title="MLflow + OpenAI Demo", version="1.0.0")




class PredictIn(BaseModel):
# Predict y for a single x or a list of x's
x: List[float]




class PredictOut(BaseModel):
y: List[float]




class ChatIn(BaseModel):
prompt: str




class ChatOut(BaseModel):
reply: str




@app.get("/")
def root():
return {"ok": True, "message": "MLflow + OpenAI demo is running."}




@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
import numpy as np


X = np.array(payload.x, dtype=float).reshape(-1, 1)
preds = model.predict(X)
return {"y": [float(v) for v in preds]}




@app.post("/chat", response_model=ChatOut)
def chat(payload: ChatIn):
# Very small wrapper around Chat Completions
res = client.chat.completions.create(
model="gpt-4o-mini",
messages=[
{"role": "system", "content": "You are a concise, helpful assistant."},
{"role": "user", "content": payload.prompt},
],
return {"reply": res.choices[0].message.content}
