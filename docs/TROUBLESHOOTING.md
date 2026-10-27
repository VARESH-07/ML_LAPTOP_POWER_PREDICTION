# Troubleshooting & Operational Guide

## Common Issues & Solutions

### 1. `pytest.exe` or subprocess blocked on Windows
- **Symptom**: `Program 'pytest.exe' failed to run: An Application Control policy has blocked this file`
- **Cause**: Windows AppLocker or Application Control policy prevents executing un-signed `.exe` binaries inside virtual environment `Scripts/`.
- **Solution**: Execute pytest via the python interpreter module:
  ```powershell
  python -m pytest -v
  ```

### 2. Discharge Power (W) is zero or missing
- **Symptom**: `discharge_power_w` reports `0.0` or `None`.
- **Cause**: Laptop is currently plugged into AC power (charging state) OR battery IC hardware does not expose instantaneous discharge rate.
- **Solution**:
  - Ensure AC charger is unplugged when testing discharge rate.
  - The application automatically falls back to calculating energy drain delta over sample intervals.

### 3. Streamlit web browser fails to open automatically
- **Symptom**: `streamlit run` prints URL but does not launch browser window.
- **Solution**: Manually open http://localhost:8501 in your browser.

### 4. GPU utilization metric is 0% or N/A
- **Symptom**: Integrated AMD Radeon 860M GPU shows fixed or estimated GPU utilization.
- **Cause**: Standard `psutil` handles CPU and RAM. Direct GPU hardware queries require vendor-specific APIs (NVML for NVIDIA, sysfs `/sys/class/drm/card*` for AMD on Linux).
- **Solution**: The application uses workload-assisted estimation when hardware GPU counters are unexposed.
