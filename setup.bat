@echo off
echo Setting up Plant Disease VQA system...

:: Create necessary directories
if not exist checkpoints mkdir checkpoints
if not exist uploads mkdir uploads

:: Check for Python
where python >nul 2>nul
if errorlevel 1 (
    echo Python is not installed. Please install Python 3 and try again.
    exit /b 1
)

:: Check for pip
where pip >nul 2>nul
if errorlevel 1 (
    echo pip is not installed. Please install pip and try again.
    exit /b 1
)

:: Create virtual environment
echo 🔧 Creating virtual environment...
:: python -m venv VQAenv
python -3.10 -m venv vqaenv310 
:: required for pip install sentencepiece

:: Activate virtual environment
echo 🔧 Activating virtual environment...
:: call VQAenv\Scripts\activate.bat
call vqaenv310\Scripts\activate.bat

:: Install dependencies
echo Installing dependencies...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers pillow fastapi uvicorn streamlit python-multipart nltk bert-score sentence-transformers sentencepiece

:: Check for checkpoint
if not exist "checkpoints\best_checkpoint_epoch_0.pth" (
    echo Checkpoint file not found. Please download the checkpoint file and place it in the checkpoints directory.
    echo Expected location: checkpoints\best_bleu_0.5439_epoch_4.pth
)

:: Start API in background
echo Starting API server...
start "API Server" cmd /c "uvicorn inference_api:app --reload --host 0.0.0.0 --port 8000"

:: Wait a bit for API to start
echo Waiting for API to start...
timeout /t 5 >nul

:: Start Streamlit app
echo Starting Streamlit frontend...
streamlit run streamlit_app.py

echo Done!
