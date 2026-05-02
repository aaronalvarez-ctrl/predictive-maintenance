import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from collections import deque
from datetime import datetime

import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

# --- Load artifacts ---
binary_pipe   = joblib.load('output/binary_pipe.pkl')
multi_pipe    = joblib.load('output/multi_pipe.pkl')
thresholds    = joblib.load('output/thresholds.pkl')
TORQUE_MAX    = joblib.load('output/torque_max.pkl')
FEATURE_NAMES = joblib.load('output/feature_names.pkl')

FAILURE_LABELS = {
    0: 'Normal Operation',
    1: 'Overheating',
    2: 'Electrical Fault',
    3: 'Overload',
    4: 'Degradation Failure',
}

FAILURE_SYMPTOMS = {
    1: {
        'info':     {'message': 'Motor temperatures are slightly above baseline.', 'trigger_stop': False},
        'warning':  {'message': 'Temperatures are rising beyond normal range.', 'trigger_stop': False},
        'critical': {'message': 'Motor is overheating critically.', 'trigger_stop': True}
    },
    2: {
        'info':     {'message': 'Minor irregularities in speed and torque.', 'trigger_stop': False},
        'warning':  {'message': 'Speed and torque are becoming erratic.', 'trigger_stop': False},
        'critical': {'message': 'Severe electrical instability detected.', 'trigger_stop': True}
    },
    3: {
        'info':     {'message': 'Torque is running slightly above nominal.', 'trigger_stop': False},
        'warning':  {'message': 'Motor is operating under significant overload.', 'trigger_stop': False},
        'critical': {'message': 'Extreme overload detected.', 'trigger_stop': True}
    },
    4: {
        'info':     {'message': 'Gradual drift detected across multiple sensors.', 'trigger_stop': False},
        'warning':  {'message': 'Progressive degradation pattern confirmed.', 'trigger_stop': False},
        'critical': {'message': 'Advanced degradation detected across thermal.', 'trigger_stop': True}
    }
}

TREND_MESSAGES = {
    'TREND_RISING': {
        'info':    {'message': 'Failure probability is slowly creeping up.', 'trigger_stop': False},
        'warning': {'message': 'Failure probability has been consistently rising.', 'trigger_stop': False}
    },
    'RUL_DROPPING': {
        'warning':  {'message': 'Remaining Useful Life is declining faster than expected.', 'trigger_stop': False},
        'critical': {'message': 'RUL is critically low.', 'trigger_stop': False}
    },
    'SUSTAINED_RISK': {
        'warning': {'message': 'Failure probability has been elevated for a sustained period.', 'trigger_stop': False}
    }
}


class MotorHealth:
    def __init__(self, motor_id, history_size=20):
        self.motor_id      = motor_id
        self.state_history = deque(maxlen=2)
        self.prob_history  = deque(maxlen=history_size)
        self.rul_history   = deque(maxlen=history_size)
        self.alerts        = []
        self.reading_count = 0
        self.last_state    = 'NORMAL'
        self.last_seen     = None
        self.shutdown      = False

    def update(self, prob, rul, state):
        self.prob_history.append(prob)
        self.rul_history.append(rul)
        self.state_history.append(state)
        self.reading_count += 1
        self.last_seen = datetime.now().isoformat()

    @property
    def prob_trend(self):
        if len(self.prob_history) < 5:
            return 0.0
        probs = list(self.prob_history)
        x = np.arange(len(probs[-5:]))
        return float(np.polyfit(x, probs[-5:], 1)[0])

    @property
    def rul_trend(self):
        if len(self.rul_history) < 5:
            return 0.0
        ruls = list(self.rul_history)
        x = np.arange(len(ruls[-5:]))
        return float(np.polyfit(x, ruls[-5:], 1)[0])

    @property
    def avg_prob(self):
        if not self.prob_history:
            return 0.0
        return float(np.mean(self.prob_history))

    def apply_persistence(self, new_state):
        self.state_history.append(new_state)
        states = list(self.state_history)
        if states[-1] == 'FAILURE':
            return 'FAILURE'
        if states.count('WARNING') >= 2:
            return 'WARNING'
        return new_state

    def get_band(self, prob):
        LOW, HIGH = thresholds['balanced']
        if prob < 0.03: return None
        if prob < LOW:  return 'info'
        if prob < HIGH: return 'warning'
        return 'critical'

    def autonomous_alerts(self, prob, rul, failure_type):
        alerts = []
        band = self.get_band(prob)

        if failure_type != 0 and band:
            symptoms = FAILURE_SYMPTOMS.get(failure_type, {}).get(band)
            if symptoms:
                key = ['', 'OVERHEATING', 'ELECTRICAL_FAULT', 'OVERLOAD', 'DEGRADATION'][failure_type]
                alerts.append({
                    'type':         f'{key}__{band.upper()}',
                    'severity':     band.upper(),
                    'message':      symptoms['message'],
                    'trigger_stop': symptoms['trigger_stop'],
                    'failure_prob': round(prob, 4),
                })

        if self.prob_trend > 0.005:
            trend_band = 'warning' if self.prob_trend > 0.01 else 'info'
            msg = TREND_MESSAGES['TREND_RISING'].get(trend_band)
            if msg:
                alerts.append({
                    'type':         f'TREND_RISING__{trend_band.upper()}',
                    'severity':     trend_band.upper(),
                    'message':      msg['message'],
                    'trigger_stop': msg['trigger_stop'],
                    'prob_trend':   round(self.prob_trend, 5),
                })

        if self.rul_trend < -1.0:
            rul_band = 'critical' if rul < 20 else 'warning'
            msg = TREND_MESSAGES['RUL_DROPPING'].get(rul_band)
            if msg:
                alerts.append({
                    'type':         f'RUL_DROPPING__{rul_band.upper()}',
                    'severity':     rul_band.upper(),
                    'message':      msg['message'],
                    'trigger_stop': msg['trigger_stop'],
                    'rul':          round(rul, 1),
                    'rul_trend':    round(self.rul_trend, 3),
                })

        if self.avg_prob > 0.05 and len(self.prob_history) >= 10:
            msg = TREND_MESSAGES['SUSTAINED_RISK']['warning']
            alerts.append({
                'type':         'SUSTAINED_RISK__WARNING',
                'severity':     'WARNING',
                'message':      msg['message'],
                'trigger_stop': msg['trigger_stop'],
                'avg_prob':     round(self.avg_prob, 4),
            })

        trigger_stop = any(a.get('trigger_stop') for a in alerts)
        self.alerts = alerts
        return alerts, trigger_stop


motor_registry = {
    f'motor_{i}': MotorHealth(f'motor_{i}') for i in range(1, 4)
}

app = FastAPI(title='Predictive Maintenance API')


class SensorReading(BaseModel):
    motor_id:            str
    Type:                str
    air_temperature:     float
    process_temperature: float
    rotational_speed:    float
    torque:              float
    tool_wear:           float
    degradation:         float
    RUL:                 float
    strategy:            Optional[str] = 'balanced'


def engineer_features(row: dict) -> dict:
    row['Temp rise']            = row['process_temperature'] - row['air_temperature']
    row['Power approx']         = row['torque'] * row['rotational_speed']
    row['Torque norm']          = row['torque'] / TORQUE_MAX
    row['Speed deviation']      = abs(row['rotational_speed'] - 1500)
    row['Degradation x Torque'] = row['degradation'] * row['torque']
    row['Temp stress']          = row['Temp rise'] / row['air_temperature']
    row['Load stress']          = row['torque'] * row['Speed deviation']
    row['Energy stress']        = row['Power approx'] * row['degradation']
    row['Combined stress']      = row['Temp rise'] + row['Torque norm'] + row['Speed deviation']
    return row

def rename_keys(row: dict) -> dict:
    base = {
        'Type':                row['Type'],
        'Air temperature':     row['air_temperature'],
        'Process temperature': row['process_temperature'],
        'Rotational speed':    row['rotational_speed'],
        'Torque':              row['torque'],
        'Tool wear':           row['tool_wear'],
        'Degradation':         row['degradation'],
        'RUL':                 row['RUL'],
    }
    for k in ['Temp rise', 'Power approx', 'Torque norm', 'Speed deviation',
              'Degradation x Torque', 'Temp stress', 'Load stress', 'Energy stress', 'Combined stress']:
        if k in row:
            base[k] = row[k]
    return base

def predict_single(data: dict, motor: MotorHealth, strategy: str = 'balanced'):
    LOW, HIGH = thresholds[strategy]
    row     = engineer_features(dict(data))
    row     = rename_keys(row)
    X_input = pd.DataFrame([row])[FEATURE_NAMES]
    prob    = float(binary_pipe.predict_proba(X_input)[0, 1])
    rul     = data.get('RUL', 0)

    if prob < LOW:
        state, failure_type, confidence = 'NORMAL', 0, 'HIGH'
    else:
        failure_type = int(multi_pipe.predict(X_input)[0])
        state        = 'WARNING' if prob < HIGH else 'FAILURE'
        confidence   = 'LOW' if failure_type == 4 and prob < 0.7 else 'HIGH'

    motor.update(prob, rul, state)
    persistent_state     = motor.apply_persistence(state)
    alerts, trigger_stop = motor.autonomous_alerts(prob, rul, failure_type)
    motor.last_state     = persistent_state

    return {
        'motor_id':      motor.motor_id,
        'state':         persistent_state,
        'failure_type':  failure_type,
        'failure_label': FAILURE_LABELS.get(failure_type, 'Unknown'),
        'confidence':    confidence,
        'failure_prob':  round(prob, 4),
        'RUL':           rul,
        'strategy':      strategy,
        'alerts':        alerts,
        'trigger_stop':  trigger_stop,
        'health': {
            'prob_trend':    round(motor.prob_trend, 5),
            'rul_trend':     round(motor.rul_trend, 3),
            'avg_prob':      round(motor.avg_prob, 4),
            'reading_count': motor.reading_count,
        },
        'timestamp': datetime.now().isoformat()
    }


@app.post('/predict')
def predict(reading: SensorReading):
    motor = motor_registry.get(reading.motor_id)
    if not motor:
        raise HTTPException(status_code=400, detail=f'Unknown motor_id: {reading.motor_id}')
    if motor.shutdown:
        raise HTTPException(status_code=403, detail=f'{reading.motor_id} is shutdown, reset it first')
    return predict_single(reading.dict(), motor, reading.strategy)

@app.post('/predict/batch')
def predict_batch(readings: list[SensorReading]):
    results = []
    for r in readings:
        motor = motor_registry.get(r.motor_id)
        if not motor:
            raise HTTPException(status_code=400, detail=f'Unknown motor_id: {r.motor_id}')
        if motor.shutdown:
            results.append({'motor_id': r.motor_id, 'state': 'SHUTDOWN', 'message': 'Motor is shutdown'})
            continue
        results.append(predict_single(r.dict(), motor, r.strategy))
    return results

@app.post('/motors/{motor_id}/stop')
def stop_motor(motor_id: str):
    motor = motor_registry.get(motor_id)
    if not motor:
        raise HTTPException(status_code=404, detail=f'Unknown motor_id: {motor_id}')
    motor.shutdown   = True
    motor.last_state = 'SHUTDOWN'
    motor.alerts     = []
    print(f'[API] {motor_id} stopped by HMI')
    return {'status': 'stopped', 'motor_id': motor_id, 'timestamp': datetime.now().isoformat()}

@app.post('/motors/{motor_id}/reset')
def reset_motor(motor_id: str):
    motor_registry[motor_id] = MotorHealth(motor_id)
    print(f'[API] {motor_id} reset by HMI')
    return {'status': 'reset', 'motor_id': motor_id, 'timestamp': datetime.now().isoformat()}

@app.get('/motors/status')
def motors_status():
    return {
        mid: {
            'state':         m.last_state,
            'shutdown':      m.shutdown,
            'avg_prob':      round(m.avg_prob, 4),
            'prob_trend':    round(m.prob_trend, 5),
            'rul_trend':     round(m.rul_trend, 3),
            'active_alerts': m.alerts,
            'reading_count': m.reading_count,
            'last_seen':     m.last_seen,
        }
        for mid, m in motor_registry.items()
    }

@app.get('/motors/{motor_id}/alerts')
def motor_alerts(motor_id: str):
    motor = motor_registry.get(motor_id)
    if not motor:
        raise HTTPException(status_code=404, detail=f'Unknown motor_id: {motor_id}')
    return {
        'motor_id':      motor_id,
        'active_alerts': motor.alerts,
        'state':         motor.last_state,
        'shutdown':      motor.shutdown,
        'avg_prob':      round(motor.avg_prob, 4),
        'rul_trend':     round(motor.rul_trend, 3),
    }

@app.get('/health')
def health():
    return {
        'status':         'ok',
        'motors':         list(motor_registry.keys()),
        'total_readings': sum(m.reading_count for m in motor_registry.values())
    }
