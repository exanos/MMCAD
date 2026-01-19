@echo off
REM Activate CLIP4CAD conda environment
echo Activating CLIP4CAD environment...
call conda activate clip4cad
echo.
echo Environment activated successfully!
echo.
echo Python version:
python --version
echo.
echo GPU information:
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo.
echo To deactivate, run: conda deactivate
