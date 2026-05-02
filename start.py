import subprocess
import sys
import time
import os

def main():
    os.makedirs('C:\\PMS', exist_ok=True)
    print('Starting Predictive Maintenance System...')

    api = subprocess.Popen(
        [sys.executable, '-m', 'uvicorn', 'serve_model:app',
         '--host', '0.0.0.0', '--port', '8000'],
        stdout=sys.stdout, stderr=sys.stderr
    )

    time.sleep(3)

    sim = subprocess.Popen(
        [sys.executable, 'simulate.py'],
        stdout=sys.stdout, stderr=sys.stderr
    )

    bridge = subprocess.Popen(
        [sys.executable, 'wincc_bridge.py'],
        stdout=sys.stdout, stderr=sys.stderr
    )

    print('System running. Press Ctrl+C to stop.')

    try:
        api.wait()
        sim.wait()
        bridge.wait()
    except KeyboardInterrupt:
        print('\nShutting down...')
        api.terminate()
        sim.terminate()
        bridge.terminate()

if __name__ == '__main__':
    main()