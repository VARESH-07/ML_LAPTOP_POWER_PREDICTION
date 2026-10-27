# Linux Setup & Deployment Guide

## System Prerequisites

Supported Linux Distributions: Ubuntu 20.04+, Debian, Arch Linux, Fedora, Linux Mint.

Recommended System Packages:
```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip git lm-sensors powerprofilesctl upower -y
```

## Setup & Execution Steps

1. **Clone & Environment Creation**:
   ```bash
   git clone <repository_url>
   cd "MACHINE LEARNING"
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Battery Telemetry Access**:
   Ensure battery sysfs path exists:
   ```bash
   ls -la /sys/class/power_supply/BAT*
   ```

4. **Run Live Telemetry Readout**:
   ```bash
   python main.py --sample
   ```

5. **Train ML Pipeline**:
   ```bash
   python train.py
   ```

6. **Launch Interactive Web Dashboard**:
   ```bash
   streamlit run laptop_power/dashboard/app.py
   ```
