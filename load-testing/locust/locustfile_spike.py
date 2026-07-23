"""
Spike-load scenario with GRADUAL, periodic surges (not instant spikes).

Each 6-min cycle: baseline (10u) -> ramp up to 180u over 90s -> peak hold (60s)
-> ramp down over 90s -> rest. The ~90s rise is forecastable by the LSTM ~90s
ahead, so the predictive autoscaler can pre-scale before the peak. Periodic so
the model can learn the pattern. (Instant spikes are unforecastable.)

Uses LoadTestShape so it works headless in-cluster.
"""
import os

from locust import HttpUser, task, between, LoadTestShape

CYCLE = 360  # 6-minute surge cycle (5 surges in a 30-min run)

# Locust ignores --run-time when a LoadTestShape is active, the shape decides
# when to stop. LOAD_SECONDS makes the duration controllable (smoke tests etc.).
STOP_SECONDS = int(os.getenv("LOAD_SECONDS", "3600"))


def spike_users(run_time: float) -> int:
    c = run_time % CYCLE
    if c < 60:                        # 0-1 min: baseline
        return 10
    if c < 150:                       # 1-2.5 min: ramp UP 10 -> 180 over 90s
        return int(10 + (180 - 10) * (c - 60) / 90)
    if c < 210:                       # 2.5-3.5 min: peak hold
        return 180
    if c < 300:                       # 3.5-5 min: ramp DOWN over 90s
        return int(180 - (180 - 10) * (c - 210) / 90)
    return 10                         # 5-6 min: rest


class SpikeShape(LoadTestShape):
    def tick(self):
        rt = self.get_run_time()
        if rt >= STOP_SECONDS:        # default 3600s; override via LOAD_SECONDS
            return None
        return (spike_users(rt), 30)


class SpikeUser(HttpUser):
    wait_time = between(0.1, 1)

    @task(4)
    def get_products(self):
        self.client.get("/api/products", name="/api/products [GET]")

    @task(4)
    def cpu_heavy(self):
        self.client.get("/api/load/cpu?iterations=1000", name="/api/load/cpu-heavy [GET]")

    @task(2)
    def memory_heavy(self):
        self.client.get("/api/load/memory?objects=15000", name="/api/load/memory-heavy [GET]")
