# Project 2.1 - Group 14 Setup Guide

## Required Software
- Unity Editor: 2022.3.62f3 LTS (Do NOT use Unity 6 / 6000.x)
- Python: 3.10.11 (via virtual environment)
- Git

## Installation Steps
1. Clone the team fork:
   git clone --branch fix-numpy-release-21-branch https://github.com/Group-14-AI-ML/ml-agents.git
2. Create and activate a Python virtual environment:
   python -m venv venv
   venv\Scripts\activate
3. Install dependencies in editable mode:
   cd ml-agents
   pip install -e ./ml-agents-envs
   pip install -e ./ml-agents
4. Apply critical version fixes:
   pip install "setuptools<70.0.0"
   pip install onnxscript --no-deps

## Verification Run
Run:
   mlagents-learn config/ppo/3DBall.yaml --run-id=test_run --force
Then open the Project in Unity Editor 2022.3.62f3, load 3DBall.unity, and hit Play.