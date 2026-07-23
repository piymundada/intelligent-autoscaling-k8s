"""
GC-pressure scenario: designed to expose the unique value of JVM-aware scaling.

Each 8-min cycle:
  - Baseline: light traffic (10u), normal requests + mild GC load
  - Ramp:     90s ramp to 80 users hitting /api/load/gc heavily
  - Peak:     sustained GC-pressure burst (80u, mostly /api/load/gc)
  - Recovery: ramp down over 60s

Why this is different from bursty/spike:
  /api/load/gc allocates 4 MB chunks × 20 cycles per request = 80 MB short-lived
  garbage per hit, forcing frequent Young GC pauses with LOW CPU usage.

  HPA (CPU 60% + Memory 70%): CPU stays ~30-40% during the burst → HPA sees
  nothing abnormal → stays at current replicas → latency degrades from GC pauses.

  LSTM + JVM floor:
    - jvm_gc_avg_pause_ms spikes above 10ms threshold → floor fires → +1 replica
    - jvm_gc_pause_count rises rapidly → LSTM learns the pattern
    - response_time_p95_ms climbs → LSTM predicts higher demand
  Together: autoscaler scales up to absorb GC pressure before SLA is breached.

  This is the scenario that proves JVM-aware scaling is INDEPENDENTLY valuable
  (not just correlated with CPU): the floor contribution is unique here.

Uses LoadTestShape so it works headless in-cluster.
"""
import os

from locust import HttpUser, task, between, LoadTestShape

CYCLE = 480   # 8-minute cycle

# Locust ignores --run-time when a LoadTestShape is active, the shape decides
# when to stop. LOAD_SECONDS makes the duration controllable (smoke tests etc.).
STOP_SECONDS = int(os.getenv("LOAD_SECONDS", "3600"))


def gc_users(run_time: float) -> int:
    c = run_time % CYCLE
    if c < 180:                         # 0-3 min: baseline
        return 10
    if c < 270:                         # 3-4.5 min: ramp UP 10→80 over 90s
        return int(10 + (80 - 10) * (c - 180) / 90)
    if c < 390:                         # 4.5-6.5 min: peak GC-pressure burst
        return 80
    if c < 450:                         # 6.5-7.5 min: ramp DOWN over 60s
        return int(80 - (80 - 10) * (c - 390) / 60)
    return 10                           # 7.5-8 min: rest


class GCShape(LoadTestShape):
    def tick(self):
        rt = self.get_run_time()
        if rt >= STOP_SECONDS:          # default 3600s; override via LOAD_SECONDS
            return None
        return (gc_users(rt), 20)


class GCUser(HttpUser):
    wait_time = between(0.5, 2)

    # Heavy GC pressure: 80 MB short-lived garbage per request, low CPU
    @task(5)
    def gc_pressure(self):
        self.client.get(
            "/api/load/gc?cycles=20&chunkKb=4096",
            name="/api/load/gc [GC-pressure]"
        )

    # Normal product reads: keeps RPS signal alive for LSTM
    @task(3)
    def get_products(self):
        self.client.get("/api/products", name="/api/products [GET]")

    # Light memory load: keeps heap utilisation visible but below HPA threshold
    @task(2)
    def memory_light(self):
        self.client.get(
            "/api/load/memory?objects=3000",
            name="/api/load/memory-light [GET]"
        )
