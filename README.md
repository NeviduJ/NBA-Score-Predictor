---
title: Nba Score Predictor
emoji: 🌖
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.1.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Overview
This app predicts an NBA team’s score for a given quarter using stats entered up to that quarter. It uses TensorFlow models specialized per quarter and serves a Gradio interface.

## Repository contents
- `app.py`: Gradio UI and prediction logic
- `requirements.txt`: Python dependencies
- `2022_2023_NBA_Season_Quarterly_Data.csv`: Reference dataset to align features/encodings
- `ANNR_ts_q1_exp1_model.keras`: Model for quarter 1
- `ANNR_ts_q2_exp1_model.keras`: Model for quarter 2
- `ANNR_ts_q3_exp1_model.keras`: Model for quarter 3

## Setup
Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

Key libraries: `gradio`, `tensorflow`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `catboost`.

## Run locally
```bash
python app.py
```
This launches a Gradio interface in your browser.

## Using the app
Fill in the following inputs (typically values accumulated up to the selected quarter), then click "Predict Score":
- `Team Name` (dropdown)
- `Quarter` (1, 2, or 3)
- `Field Goals Made (FGM)`
- `Field Goals Attempted (FGA)`
- `3 Pointers Made (3PM)`
- `3 Pointers Attempted (3PA)`
- `Free Throws Made (FTM)`
- `Free Throws Attempted (FTA)`
- `Rebounds Offensive (OREB)`
- `Rebounds Defensive (DREB)`
- `Rebounds Total (REB)`
- `Assists (AST)`
- `Steals (STL)`
- `Blocks (BLK)`
- `Turnovers (TO)`
- `Personal Fouls (PF)`
- `Points (PTS)`
- `+/- Points (+/-)`

The output textbox shows the predicted score for the selected quarter.

## Model selection
`app.py` loads a model based on the `Quarter` value:
- Q1 → `ANNR_ts_q1_exp1_model.keras`
- Q2 → `ANNR_ts_q2_exp1_model.keras`
- Q3 → `ANNR_ts_q3_exp1_model.keras`

## Notes
- The dataset is used to construct and align one-hot encoded features to match model expectations (not for training at runtime).
- Negative predictions are clipped to 0.
- CPU is sufficient; TensorFlow-GPU is not required.

## Deployment (Hugging Face Spaces)
This repo contains Spaces metadata targeting `app.py` (Gradio SDK `5.1.0`). Pushing this repository to a Space will auto-run the app.
