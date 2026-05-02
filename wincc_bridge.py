import requests
import time
import json
import os

API = 'http://localhost:8000'
OUTPUT_FILE = 'motor_status.json'  # WinCC reads this

def poll():
    while True:
        try:
            r = requests.get(f'{API}/motors/status', timeout=2)
            data = r.json()
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f'Bridge error: {e}')
        time.sleep(1)

if __name__ == '__main__':
    poll()