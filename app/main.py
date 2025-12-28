from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from PIL import Image
import torch
import uuid 
from pathlib import Path
from app.model import load_models, predict
from app.preprocess import transform
from app.visualize import plot_prediction
from app.labels import LABELS
from app.thresholds import THRESHOLDS
import matplotlib.pyplot as plt
import base64
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

app = FastAPI()
templates = Jinja2Templates(directory="templates")
models = load_models()


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>X-ray Classifier</title>
        </head>
        <body>
            <h2>Upload Chest X-ray</h2>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input type="file" name="file" required><br><br>

                <label>Sex:</label>
                <select name="sex">
                    <option value="0">Male</option>
                    <option value="1">Female</option>
                </select><br><br>

                <label>Age:</label>
                <input type="number" name="age" step="1" value="0"><br><br>

                <label>Frontal / Lateral:</label>
                <select name="frontal_lateral">
                    <option value="0">Frontal</option>
                    <option value="1">Lateral</option>
                </select><br><br>

                <label>AP / PA:</label>
                <select name="ap_pa">
                    <option value="0">AP</option>
                    <option value="1">PA</option>
                </select><br><br>

                <button type="submit">Predict</button>
            </form>
        </body>
    </html>
    """


@app.post("/predict")
async def predict_xray(
    request: Request,
    file: UploadFile = File(...),
    sex: int = Form(...),
    age: float = Form(...),
    frontal_lateral: int = Form(...),
    ap_pa: int = Form(...)
):
    img = Image.open(file.file).convert("RGB")
    image_tensor = transform(img).unsqueeze(0)

    age_norm = age / 90.0
    sex_label = "Male" if sex == 0 else "Female"

    tabular = torch.tensor(
        [[sex, age_norm, frontal_lateral, ap_pa]],
        dtype=torch.float32
    )

    probs = predict(models, image_tensor, tabular)

    buf = plot_prediction(img, probs, LABELS, THRESHOLDS)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    results = []
    for p, t in zip(probs, THRESHOLDS):
        diff = THRESHOLDS[t] - p
        l = ""
        # pred pos -> t < p

        if diff < -0.01:
            l = "Positive"
        elif diff > 0.05:
            l = "Negative"
        else:
            l = "Further testing required"
        results.append(l)

    results = list(zip(LABELS, results))
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "plot_b64": img_b64,
            "results": results,
            "labels": LABELS,
            "age": age,
            "sex": sex_label
        }
    )