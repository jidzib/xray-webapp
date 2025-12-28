# Multimodal Chest X-ray Classification Web App

**Live Demo:** https://xray-webapp.fly.dev 

This product is a deployed web app for multimodal chest X-ray analysis. It 
combines image-based deep learning with patient tabular features to produce
interpretable diangostic predictions and exportable reports.

The app is designed as an educational demo showcasing end-to-end ML depployment,
including model inference, visualization, and user-facing reporting.

#### Key Features
    - Multimodal inference (X-ray image + patient metadata)
    - Threshold-aware predictions for more accurate results
    - Visualization of model confidence vs decision thresholds
    - Client-side export of prediction reports (no server-side image storage)
    - Fully containerized deployment using Docker and Fly.io

#### System Architecture
    - Backend: FastAPI
    - Inference: PyTorch (CPU-only)
    - Visualization: Matplotlib
    - Frontend: Server-rendered HTML using Jinja2 templates
    - Deployment: Docker + Fly.io

#### Model Notes 
The underlying model was trained on the CheXpert-small dataset and is intended
for educational and demonstration purposes only.

This system is **not** clinically validated and must not be used for medical diagnosis

### Running Locally

#### 1. Create a virtual environment
python -m venv venv
source venv/bin/activate # or venv/Scripts/activate for Windows

#### 2. Install dependencies
pip install -r requirements.txt

#### 3. Run the app
uvicorn app.main:app --reload

#### Deployment
The application is deployed using Docker and Fly.io The Dockerfile defines the runtime
environment and launches the FastAPI app via Uvicorn on port 8080.

**Training notebooks and model development code are available here**: https://github.com/jidzib/xray-model-training


**DISCLAIMER (IMPORTANT)**
## THIS PROJECT IS FOR EDUCATIONAL PURPOSES ONLY AND IS NOT INTENDED FOR CLINICAL OR DIAGNOSTIC USE