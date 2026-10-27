# Windows Setup & Compatibility Guide

## System Environment & Requirements

Supported OS: Windows 10, Windows 11 (tested on ASUS Vivobook 14 AMD Ryzen AI 7 350).

Prerequisites:
- Python 3.11+
- PowerShell 5.1+ or PowerShell Core 7+
- Windows Management Instrumentation (WMI) / CIM

## Setup Commands (PowerShell)

1. **Create & Activate Virtual Environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies**:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. **Verify Windows Battery Telemetry**:
   Test CIM/WMI battery status in PowerShell:
   ```powershell
   Get-CimInstance -ClassName Win32_Battery
   ```

4. **Run Sample Telemetry & Prediction Readout**:
   ```powershell
   .\.venv\Scripts\python.exe main.py --sample
   ```

5. **Train Model Pipeline**:
   ```powershell
   .\.venv\Scripts\python.exe train.py
   ```

6. **Run Automated Test Suite**:
   ```powershell
   .\.venv\Scripts\python.exe -m pytest -v
   ```

7. **Launch Web Dashboard**:
   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run laptop_power/dashboard/app.py
   ```
