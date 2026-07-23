"""
Bursty-load scenario with GRADUAL ramps (not instant steps).

Each 10-min cycle: low (20u) -> ramp up to 180u over 120s -> high (180u) ->
ramp down to 20u over 60s. The gradual rise gives the LSTM a leading signal it
can actually forecast ~90s ahead, so predictive scaling is meaningful (instant
step changes are unforecastable and collapse the model to mean-prediction).

Uses LoadTestShape so it works headless in-cluster.
"""
import os

from locust import HttpUser, task, between, LoadTestShape

CYCLE = 600  # 10-minute cycle (3 cycles in a 30-min run)

# Locust ignores --run-time when a LoadTestShape is active, the shape decides
# when to stop. LOAD_SECONDS makes the duration controllable (smoke tests etc.).
STOP_SECONDS = int(os.getenv("LOAD_SECONDS", "3600"))


def bursty_users(run_time: float) -> int:
    c = run_time % CYCLE
    if c < 240:                       # 0-4 min: low
        return 20
    if c < 360:                       # 4-6 min: ramp UP 20 -> 180 over 120s
        return int(20 + (180 - 20) * (c - 240) / 120)
    if c < 540:                       # 6-9 min: high
        return 180
    return int(180 - (180 - 20) * (c - 540) / 60)  # 9-10 min: ramp DOWN over 60s


class BurstyShape(LoadTestShape):
    def tick(self):
        rt = self.get_run_time()
        if rt >= STOP_SECONDS:        # default 3600s; override via LOAD_SECONDS
            return None
        return (bursty_users(rt), 30)


class BurstyUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(4)
    def get_products(self):
        self.client.get("/api/products", name="/api/products [GET]")

    @task(3)
    def cpu_load(self):
        self.client.get("/api/load/cpu?iterations=1000", name="/api/load/cpu [GET]")

    @task(3)
    def memory_load(self):
        self.client.get("/api/load/memory?objects=15000", name="/api/load/memory [GET]")
