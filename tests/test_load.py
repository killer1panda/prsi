"""
Load testing configuration using Locust.
Simulates realistic traffic patterns for the Doom Index API.
"""
from locust import HttpUser, task, between, events
import random
import json


class DoomIndexUser(HttpUser):
    """Simulated user interacting with Doom Index API."""

    wait_time = between(1, 5)  # Wait 1-5 seconds between tasks
    weight = 10  # 10x more common than admin users

    def on_start(self):
        """Setup per user session."""
        self.sample_texts = [
            "I think this celebrity made a huge mistake and should apologize.",
            "Just had the best coffee ever! ☕ #blessed",
            "This politician is corrupt and needs to be investigated immediately!!!",
            "Happy birthday to my best friend! 🎉",
            "The new policy is completely unfair to working families.",
            "Cute cat video thread 🐱",
            "How dare they say such things in public? Cancel them!",
            "Beautiful sunset today.",
            "This company exploits workers and should be boycotted.",
            "My thoughts on the latest movie release..."
        ]

    @task(5)
    def analyze_single_post(self):
        """Most common task: analyze a single post."""
        payload = {
            "text": random.choice(self.sample_texts),
            "source": random.choice(["reddit", "twitter", "instagram"]),
            "user_id": f"user_{random.randint(1, 10000)}"
        }

        with self.client.post("/analyze", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "doom_score" not in data:
                    response.failure("Missing doom_score in response")
                elif not (0 <= data["doom_score"] <= 100):
                    response.failure(f"Invalid doom_score: {data['doom_score']}")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(3)
    def analyze_batch(self):
        """Batch prediction for power users."""
        payload = {
            "items": [
                {"text": random.choice(self.sample_texts), "user_id": f"u{i}"}
                for i in range(random.randint(5, 20))
            ]
        }

        with self.client.post("/predict/batch", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if len(data.get("predictions", [])) != len(payload["items"]):
                    response.failure("Batch size mismatch")

    @task(2)
    def attack_simulation(self):
        """Attack simulator usage."""
        payload = {
            "text": "I think this is a reasonable opinion.",
            "strategy": random.choice(["semantic", "character", "emoji"]),
            "num_variants": random.randint(1, 5)
        }

        self.client.post("/attack/simulate", json=payload)

    @task(2)
    def check_health(self):
        """Health check polling."""
        self.client.get("/health")

    @task(1)
    def get_leaderboard(self):
        """View leaderboard."""
        self.client.get(f"/dashboard/leaderboard?limit={random.randint(5, 50)}")

    @task(1)
    def get_drift_status(self):
        """Check drift status."""
        self.client.get("/dashboard/drift-status")


class AdminUser(HttpUser):
    """Admin user performing maintenance tasks."""

    wait_time = between(10, 30)
    weight = 1

    @task(3)
    def trigger_retraining(self):
        """Trigger model retraining."""
        self.client.post("/admin/retrain", json={"model_type": "distilbert"})

    @task(2)
    def get_metrics(self):
        """Get Prometheus metrics."""
        self.client.get("/metrics")

    @task(1)
    def update_feature_store(self):
        """Update feature store materialization."""
        self.client.post("/admin/materialize", json={"feature_view": "user_features"})


# Custom event handlers for detailed reporting
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, 
               context, exception, **kwargs):
    """Log slow requests for analysis."""
    if response_time > 5000:  # 5 seconds
        print(f"SLOW REQUEST: {request_type} {name} took {response_time}ms")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Generate summary report when load test stops."""
    stats = environment.runner.stats
    print("\n" + "="*50)
    print("LOAD TEST SUMMARY")
    print("="*50)
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Failed requests: {stats.total.num_failures}")
    print(f"Avg response time: {stats.total.avg_response_time:.2f}ms")
    print(f"95th percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms")
    print(f"RPS: {stats.total.total_rps:.2f}")
    print("="*50)
