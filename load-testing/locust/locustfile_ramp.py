"""
Saturation-ramp scenario: steadily increases load on a SINGLE pinned pod until
its p95 latency approaches the SLA (300 ms). Used once, offline, to calibrate
RPS_PER_REPLICA: the sustainable requests/second a single replica can serve
within SLA. Not part of the HPA-vs-predictive comparison.

Run with the HPA deleted and the deployment pinned to 1 replica so the pod
cannot scale out, then feed the merged CSV to ml/calibrate_capacity.py (which
keeps only SLA-compliant intervals and reports per-replica RPS).

Uses the heavier spike-like task mix so the estimate is conservative.

Usage:
    kubectl -n thesis delete hpa spring-boot-hpa
    kubectl -n thesis scale deployment/spring-boot-app --replicas=1
    python3 -m locust -f locustfile_ramp.py --host=http://localhost:8080 \
           --headless --csv=data/calib_ramp --csv-full-history
"""
from locust import HttpUser, task, between, LoadTestShape


# (end_time_seconds, target_users, spawn_rate), steadily ramp up to saturation
RAMP_STAGES = [
    ( 120,    5,   5),
    ( 240,   15,   5),
    ( 360,   30,  10),
    ( 480,   50,  10),
    ( 600,   75,  15),
    ( 720,  100,  20),
    ( 840,  130,  20),
]


class RampShape(LoadTestShape):
    def tick(self):
        run_time = self.get_run_time()
        for end_time, users, spawn_rate in RAMP_STAGES:
            if run_time < end_time:
                return (users, spawn_rate)
        return None  # stop after final stage


class RampUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(4)
    def get_products(self):
        self.client.get("/api/products", name="/api/products [GET]")

    @task(4)
    def cpu_heavy(self):
        self.client.get("/api/load/cpu?iterations=1000", name="/api/load/cpu-heavy [GET]")

    @task(2)
    def memory_heavy(self):
        self.client.get("/api/load/memory?objects=15000", name="/api/load/memory-heavy [GET]")
