import time
from core.drowsiness_logic import DrowsinessMonitor

monitor = DrowsinessMonitor(threshold_seconds=3.0)

print("--- Testing Drowsiness Logic ---")
print(f"Initial status: {monitor.status}")

# Simulate awake
status = monitor.update_state("Open", "Open")
print(f"Both Open -> Status: {status}, Timer: {monitor.elapsed_closed_time:.1f}s")

# Simulate closing eyes
print("\n--- Closing Eyes ---")
for i in range(1, 5):
    time.sleep(1) # simulate 1 second passing
    status = monitor.update_state("Closed", "Closed")
    print(f"Eyes Closed ({i}s) -> Status: {status}, Timer: {monitor.elapsed_closed_time:.1f}s")

# Simulate waking up
print("\n--- Waking Up ---")
status = monitor.update_state("Open", "Open")
print(f"Both Open -> Status: {status}, Timer: {monitor.elapsed_closed_time:.1f}s")

# Simulate blinking or noise (one eye closed)
print("\n--- Simulating Noise/Blink (One eye open, one closed) ---")
time.sleep(1)
status = monitor.update_state("Closed", "Open")
print(f"Left Closed, Right Open -> Status: {status}, Timer: {monitor.elapsed_closed_time:.1f}s")
