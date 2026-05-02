# Predictive Maintenance Classification System

A machine learning system for real-time fault detection and failure classification in industrial induction motors. The system integrates a two-stage AI pipeline with a REST API, a WinCC RT Professional HMI, and a TIA Portal / S7-1500 PLC running Structured Text control logic.

---

## Overview

Predictive maintenance is one of the most impactful applications of machine learning in industrial settings. Rather than reacting to failures after they happen or replacing components on fixed schedules, this system continuously monitors motor health and classifies developing faults before they cause unplanned downtime.

The system was developed against the [Machine Predictive Maintenance Classification Dataset](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification), a synthetic dataset modeled after real industrial operating conditions covering five operational states across 10,000 data points.

---

## How It Works

Three induction motors are monitored in real time. Each motor sends sensor readings, air temperature, process temperature, rotational speed, torque, tool wear and degradation, to a FastAPI inference server every second. The server runs a two-stage machine learning pipeline: the first stage determines whether a failure is occurring, and the second classifies what type of failure it is. Results are returned with calibrated probabilities, a remaining useful life estimate, and a set of contextual alerts describing what is developing and what to check.

The HMI, built in WinCC RT Professional, displays live motor status, sensor values, RUL trends and active alerts for all three motors on a single overview screen. Operators can drill into individual motors for full detail. The PLC logic handles start/stop latching.

---

## Machine Learning Pipeline

### Stage 1 — Binary Failure Detection

A `LightGBMClassifier` wrapped in `CalibratedClassifierCV` produces calibrated failure probabilities. Calibration matters here because the downstream threshold logic depends on probabilities being meaningful, not just ranked. The 96/4 class imbalance is handled via `class_weight='balanced'`.

### Stage 2 — Failure Type Classification

Trained exclusively on failure rows, this model never sees normal operation. `SMOTE` oversamples minority classes before fitting and `Degradation Failure` receives an additional weight boost due to its low sample count. An `imblearn` pipeline ensures SMOTE is applied correctly inside cross-validation without leaking into the test set.

### Threshold Autotuning

Rather than fixed decision thresholds, the system grid-searches LOW and HIGH probability cutoffs across the test set to find the combination that optimizes the chosen strategy, either balanced Macro F1 or minimized missed failures on critical classes.

### Persistence & Alerts

A sliding window of two consecutive readings smooths noisy predictions before confirming a state change. The alert engine monitors rolling probability trends, RUL trajectory and sustained elevated risk independently of individual predictions, catching deteriorating conditions before they cross the detection threshold.

---

## Model Performance

### Binary Model — Failure Detection

| Metric | Score |
|---|---|
| ROC AUC | 0.9831 |
| Average Precision | 0.8983 |
| Matthews Correlation Coefficient | 0.8482 |
| Balanced Accuracy | 0.8905 |
| Macro F1 | 0.9225 |

### Multiclass Model — Failure Classification

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Overheating | 1.00 | 1.00 | 1.00 |
| Electrical Fault | 0.96 | 1.00 | 0.98 |
| Overload | 0.95 | 1.00 | 0.97 |
| Degradation Failure | 1.00 | 0.82 | 0.90 |

### Full System

| Strategy | Macro F1 | Balanced Accuracy | MCC |
|---|---|---|---|
| Balanced | 0.7702 | 0.8291 | 0.7463 |
| No Miss | 0.7089 | 0.8463 | 0.6176 |

> The gap between individual model scores and full system scores is expected, the two-stage threshold logic introduces additional error surface. A missed detection in Stage 1 means Stage 2 never gets to classify it. This is a deliberate tradeoff that prioritizes real-world safety behavior over benchmark numbers.

**Stability across 5 random seeds:** Mean Macro F1 `0.7557` | Std `0.0235`

---

## Project Structure

```
predictive-maintenance/
├── output/
│   ├── binary_pipe.pkl         # Stage 1 model
│   ├── multi_pipe.pkl          # Stage 2 model
│   ├── thresholds.pkl          # Auto-tuned thresholds
│   ├── torque_max.pkl          # Normalization constant
│   └── feature_names.pkl       # Column order for inference
├── pdwml.ipynb                 # Training notebook
├── serve_model.py              # FastAPI inference server
├── simulate.py                 # Motor simulator
├── wincc_bridge.py             # Writes motor_status.json for WinCC
└── start.py                    # Starts all processes
```

---

## Setup & Running

**Install dependencies**
```bash
pip install fastapi uvicorn lightgbm imbalanced-learn scikit-learn pandas numpy joblib requests
```

**Train the model**

Open `pdwml.ipynb` and run all cells. Artifacts are saved to `output/`.

**Start the system**
```bash
python start.py
```

Starts the API on `http://localhost:8000`, the motor simulator and the WinCC bridge simultaneously.

**API endpoints**
```
POST /predict                   Single motor reading
POST /predict/batch             Multiple readings
GET  /motors/status             All motors summary
GET  /motors/{id}/alerts        Active alerts per motor
POST /motors/{id}/stop          HMI stop command
POST /motors/{id}/reset         HMI reset after maintenance
GET  /health                    API health check
```