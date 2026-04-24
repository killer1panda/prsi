"""
Kafka-based real-time streaming pipeline for social media posts.
Consumes raw posts, enriches with features, and produces predictions.
"""
import logging
import json
import signal
import sys
from typing import Dict, Callable, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import torch
from confluent_kafka import Consumer, Producer, KafkaError, KafkaException

logger = logging.getLogger(__name__)


@dataclass
class KafkaConfig:
    bootstrap_servers: str = "localhost:9092"
    consumer_group: str = "doom-index-consumer"
    input_topic: str = "social-posts-raw"
    output_topic: str = "doom-predictions"
    dlq_topic: str = "doom-dlq"
    auto_offset_reset: str = "latest"
    max_poll_records: int = 100
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 10000


class KafkaPipeline:
    """
    Production Kafka consumer-producer for real-time Doom Index inference.
    Handles backpressure, dead-letter queue, and graceful shutdown.
    """

    def __init__(self, predictor: Callable, config: Optional[KafkaConfig] = None):
        self.predictor = predictor
        self.config = config or KafkaConfig()
        self.running = False

        self.consumer = Consumer({
            "bootstrap.servers": self.config.bootstrap_servers,
            "group.id": self.config.consumer_group,
            "auto.offset.reset": self.config.auto_offset_reset,
            "max.poll.interval.ms": self.config.session_timeout_ms,
            "heartbeat.interval.ms": self.config.heartbeat_interval_ms,
            "enable.auto.commit": True,
            "auto.commit.interval.ms": 5000,
        })

        self.producer = Producer({
            "bootstrap.servers": self.config.bootstrap_servers,
            "compression.type": "lz4",
            "batch.size": 16384,
            "linger.ms": 5,
            "retries": 3,
            "retry.backoff.ms": 1000
        })

        self.consumer.subscribe([self.config.input_topic])

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"KafkaPipeline initialized: consumer_group={self.config.consumer_group}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def _delivery_callback(self, err, msg):
        """Producer delivery callback."""
        if err:
            logger.error(f"Message delivery failed: {err}")

    def _send_dlq(self, message: Dict, error: str):
        """Send failed message to dead-letter queue."""
        dlq_msg = {
            "original": message,
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
            "topic": self.config.input_topic
        }
        self.producer.produce(
            self.config.dlq_topic,
            key=str(message.get("post_id", "unknown")),
            value=json.dumps(dlq_msg),
            callback=self._delivery_callback
        )

    def _process_message(self, msg_value: str) -> Optional[Dict]:
        """
        Process a single Kafka message.

        Returns:
            Prediction result dict or None if processing failed
        """
        try:
            post = json.loads(msg_value)

            # Validate required fields
            required = ["text", "user_id", "post_id"]
            if not all(k in post for k in required):
                raise ValueError(f"Missing required fields: {[k for k in required if k not in post]}")

            # Run prediction
            result = self.predictor(post)

            # Enrich with metadata
            prediction = {
                "post_id": post["post_id"],
                "user_id": post["user_id"],
                "doom_score": float(result.get("doom_score", 0)),
                "risk_level": result.get("risk_level", "unknown"),
                "timestamp": datetime.utcnow().isoformat(),
                "model_version": result.get("model_version", "unknown"),
                "features": result.get("features", {})
            }

            return prediction

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            self._send_dlq({"raw": msg_value}, f"JSON decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"Processing error: {e}")
            self._send_dlq(json.loads(msg_value) if msg_value else {}, f"Processing error: {e}")
            return None

    def run(self):
        """Main consumer loop."""
        self.running = True
        logger.info(f"Starting Kafka consumer on topic: {self.config.input_topic}")

        try:
            while self.running:
                msg = self.consumer.poll(timeout=1.0)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.debug(f"End of partition reached: {msg.topic()}[{msg.partition()}]")
                    else:
                        raise KafkaException(msg.error())
                    continue

                # Process message
                prediction = self._process_message(msg.value().decode("utf-8"))

                if prediction:
                    self.producer.produce(
                        self.config.output_topic,
                        key=str(prediction["post_id"]),
                        value=json.dumps(prediction),
                        callback=self._delivery_callback
                    )

                # Flush producer periodically
                self.producer.poll(0)

        except Exception as e:
            logger.error(f"Fatal error in Kafka loop: {e}")

        finally:
            self.shutdown()

    def shutdown(self):
        """Graceful shutdown."""
        self.running = False
        logger.info("Flushing producer and closing consumer...")
        self.producer.flush()
        self.consumer.close()
        logger.info("KafkaPipeline shutdown complete")

    def health_check(self) -> Dict[str, Any]:
        """Check Kafka connection health."""
        try:
            # Try to get metadata
            metadata = self.consumer.list_topics(timeout=5)
            return {
                "status": "healthy",
                "brokers": len(metadata.brokers),
                "topics": [t for t in metadata.topics.keys()],
                "subscribed": self.config.input_topic
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
