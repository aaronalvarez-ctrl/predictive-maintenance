import requests
import time
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

API = 'http://localhost:8000'

class MotorState(Enum):
    NORMAL    = 'NORMAL'
    DEGRADING = 'DEGRADING'
    WARNING   = 'WARNING'
    FAILURE   = 'FAILURE'
    SHUTDOWN  = 'SHUTDOWN'

class FailureMode(Enum):
    NONE             = 0
    OVERHEATING      = 1
    ELECTRICAL_FAULT = 2
    OVERLOAD         = 3
    DEGRADATION      = 4

@dataclass
class MotorConfig:
    motor_id:           str
    product_type:       str   = 'M'
    degradation_rate:   float = 0.3
    failure_rate:       float = 0.5
    fault_trigger_prob: float = 0.002
    possible_failures:  list  = field(default_factory=lambda: [1, 2, 3, 4])
    recovery_steps:     int   = 50

class MotorSimulator:
    def __init__(self, config: MotorConfig, seed=None):
        self.cfg              = config
        self.rng              = np.random.default_rng(seed)
        self.state            = MotorState.NORMAL
        self.failure_mode     = FailureMode.NONE
        self.degradation      = 0.0
        self.tool_wear        = 0.0
        self.fault_progress   = 0.0
        self.recovery_counter = 0
        self.time_step        = 0
        self.paused           = False  # controlled by HMI via API

    def check_api_state(self):
        try:
            r = requests.get(f'{API}/motors/status', timeout=2)
            status = r.json().get(self.cfg.motor_id, {})
            self.paused = status.get('shutdown', False)
        except:
            pass  # if API unreachable, keep current state

    def _base_readings(self):
        air_temp     = 300 + self.rng.normal(0, 2)
        process_temp = air_temp + 10 + self.rng.normal(0, 1)
        rot_speed    = 1500 + self.rng.normal(0, 80)
        torque       = 40 + self.rng.normal(0, 8)
        self.tool_wear += {'L': 2, 'M': 3, 'H': 5}[self.cfg.product_type]
        return air_temp, process_temp, rot_speed, torque

    def _apply_failure_signature(self, air_temp, process_temp, rot_speed, torque):
        p = self.fault_progress
        if self.failure_mode == FailureMode.OVERHEATING:
            air_temp     += p * self.rng.uniform(5, 15)
            process_temp += p * self.rng.uniform(10, 25)
            rot_speed    -= p * self.rng.uniform(0, 100)
        elif self.failure_mode == FailureMode.ELECTRICAL_FAULT:
            rot_speed += p * self.rng.normal(0, 300)
            torque    += p * self.rng.uniform(10, 30)
        elif self.failure_mode == FailureMode.OVERLOAD:
            torque    += p * self.rng.uniform(20, 40)
            rot_speed -= p * self.rng.uniform(200, 500)
        elif self.failure_mode == FailureMode.DEGRADATION:
            air_temp     += p * self.rng.uniform(2, 8)
            process_temp += p * self.rng.uniform(3, 10)
            torque       += p * self.rng.uniform(5, 15)
            rot_speed    -= p * self.rng.uniform(50, 150)
        return air_temp, process_temp, rot_speed, torque

    def _transition(self):
        if self.state == MotorState.SHUTDOWN:
            self.recovery_counter += 1
            if self.recovery_counter >= self.cfg.recovery_steps:
                self._reset()
            return

        if self.state == MotorState.NORMAL:
            if self.rng.random() < self.cfg.fault_trigger_prob:
                self.state        = MotorState.DEGRADING
                self.failure_mode = FailureMode(int(self.rng.choice(self.cfg.possible_failures)))
                print(f'[{self.cfg.motor_id}] Developing: {self.failure_mode.name}')

        elif self.state == MotorState.DEGRADING:
            self.fault_progress += self.cfg.degradation_rate * float(self.rng.uniform(0.5, 1.5)) / 100
            self.degradation    += self.cfg.degradation_rate * float(self.rng.uniform(0.5, 1.5))
            if self.fault_progress >= 0.3:
                self.state = MotorState.WARNING
                print(f'[{self.cfg.motor_id}] WARNING — {self.failure_mode.name}')

        elif self.state == MotorState.WARNING:
            self.fault_progress += self.cfg.failure_rate * float(self.rng.uniform(0.5, 1.5)) / 100
            self.degradation    += self.cfg.failure_rate * float(self.rng.uniform(0.5, 1.5))
            if self.fault_progress >= 1.0:
                self.state = MotorState.FAILURE
                print(f'[{self.cfg.motor_id}] FAILURE — {self.failure_mode.name}')

        elif self.state == MotorState.FAILURE:
            self.state            = MotorState.SHUTDOWN
            self.recovery_counter = 0
            print(f'[{self.cfg.motor_id}] SHUTDOWN')

    def _reset(self):
        self.state            = MotorState.NORMAL
        self.failure_mode     = FailureMode.NONE
        self.fault_progress   = 0.0
        self.recovery_counter = 0
        print(f'[{self.cfg.motor_id}] Recovered')

    def step(self):
        self.time_step += 1
        self.check_api_state()

        if self.paused:
            print(f'[{self.cfg.motor_id}] Paused by HMI')
            return None

        self._transition()

        if self.state == MotorState.SHUTDOWN:
            return None

        air_temp, process_temp, rot_speed, torque = self._base_readings()

        if self.state not in (MotorState.NORMAL,):
            air_temp, process_temp, rot_speed, torque = self._apply_failure_signature(
                air_temp, process_temp, rot_speed, torque
            )

        return {
            'motor_id':            self.cfg.motor_id,
            'Type':                self.cfg.product_type,
            'air_temperature':     round(float(air_temp), 2),
            'process_temperature': round(float(process_temp), 2),
            'rotational_speed':    round(float(max(800, rot_speed)), 2),
            'torque':              round(float(max(0, torque)), 2),
            'tool_wear':           round(float(self.tool_wear), 2),
            'degradation':         round(float(self.degradation), 2),
            'RUL':                 round(float(max(0, 250 - self.degradation)), 2),
            'strategy':            'balanced'
        }


def run():
    motors = [
        MotorSimulator(MotorConfig(
            motor_id='motor_1', product_type='L',
            degradation_rate=0.2, failure_rate=0.3,
            fault_trigger_prob=0.002, possible_failures=[1, 2]
        ), seed=42),
        MotorSimulator(MotorConfig(
            motor_id='motor_2', product_type='M',
            degradation_rate=0.5, failure_rate=0.6,
            fault_trigger_prob=0.003, possible_failures=[1, 2, 3, 4]
        ), seed=7),
        MotorSimulator(MotorConfig(
            motor_id='motor_3', product_type='H',
            degradation_rate=0.9, failure_rate=1.0,
            fault_trigger_prob=0.005, possible_failures=[3, 4]
        ), seed=21),
    ]

    print('Simulator running')

    while True:
        for motor in motors:
            data = motor.step()

            if data is None:
                continue

            try:
                r = requests.post(f'{API}/predict', json=data, timeout=2)
                result = r.json()
                print(
                    f"[{data['motor_id']}] "
                    f"State: {result['state']:<10} "
                    f"| {result['failure_label']:<22} "
                    f"| prob={result['failure_prob']:.3f} "
                    f"| RUL={result['RUL']} "
                    f"| alerts={len(result['alerts'])}"
                )
            except requests.exceptions.ConnectionError:
                print('API unreachable')
            except Exception as e:
                print(f"Error: {e}")

        time.sleep(1)

if __name__ == '__main__':
    run()