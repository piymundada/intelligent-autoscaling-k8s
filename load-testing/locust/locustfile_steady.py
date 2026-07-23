"""
Steady-load scenario: constant request rate for 30 minutes.
Usage:
    locust -f locustfile_steady.py --host=http://<SERVICE_URL> \
           --users 20 --spawn-rate 2 --run-time 30m --headless
"""
from locust import HttpUser, task, between, constant_throughput


class SteadyUser(HttpUser):
    # ~1 request per second per user at 20 users = ~20 RPS steady
    wait_time = constant_throughput(1)

    @task(5)
    def get_products(self):
        self.client.get("/api/products", name="/api/products [GET]")

    @task(3)
    def get_product_by_id(self):
        self.client.get("/api/products/1", name="/api/products/{id} [GET]")

    @task(2)
    def mixed_load(self):
        # Generates JVM heap + CPU pressure (heavier params so load stresses the
        # smaller 300m pods; uniform per-request cost with bursty/spike).
        self.client.get(
            "/api/load/mixed?iterations=1000&objects=15000",
            name="/api/load/mixed [GET]"
        )
