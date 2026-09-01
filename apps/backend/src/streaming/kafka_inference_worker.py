"""
Kafka Inference Worker for Doom Index

Consumes messages from the social_ingestion topic, processes them (mock inference),
and produces results to the inference_results topic. Failed messages are sent to a DLQ.
"""

import json
import logging
import signal
import time
import uuid
from typing import Any, Dict

from confluent_kafka import (Consumer, KafkaError, KafkaException, Message,
                             Producer)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("KafkaInferenceWorker")


class DoomInferenceWorker:
    """Worker class for consuming social data and producing doom predictions."""

    def __init__(
        self,
        bootstrap_servers: str,
        group_id: str,
        input_topic: str = "social_ingestion",
        output_topic: str = "inference_results",
        dlq_topic: str = "inference_dlq",
    ):
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.dlq_topic = dlq_topic

        # Initialize Consumer
        self.consumer = Consumer(
            {
                "bootstrap.servers": self.bootstrap_servers,
                "group.id": group_id,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            }
        )

        # Initialize Producer
        self.producer = Producer(
            {"bootstrap.servers": self.bootstrap_servers, "client.id": "doom-inference-producer"}
        )

        self.running = False

    def start(self) -> None:
        """Start the consumer loop."""
        self.consumer.subscribe([self.input_topic])
        self.running = True
        logger.info(f"Subscribed to topic: {self.input_topic}")

        try:
            while self.running:
                msg = self.consumer.poll(timeout=1.0)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition event
                        logger.debug(
                            f"{msg.topic()} [{msg.partition()}] reached end at offset {msg.offset()}"
                        )
                    else:
                        raise KafkaException(msg.error())
                else:
                    self._process_message(msg)

        except KeyboardInterrupt:
            logger.info("Received stop signal (KeyboardInterrupt)")
        except Exception as e:
            logger.error(f"Error in consumer loop: {e}", exc_info=True)
        finally:
            self._shutdown()

    def stop(self) -> None:
        """Stop the worker."""
        self.running = False

    def _process_message(self, msg: Message) -> None:
        """Process a single Kafka message."""
        try:
            # Decode payload
            value = msg.value()
            if not value:
                raise ValueError("Empty message payload")

            payload_str = value.decode("utf-8")
            data = json.loads(payload_str)

            # Perform inference
            result = self._predict_doom(data)

            # Publish result
            self._publish_result(result)

            # Commit offset on success
            self.consumer.commit(asynchronous=False)
            logger.debug(f"Successfully processed and committed message at offset {msg.offset()}")

        except Exception as e:
            logger.error(f"Failed to process message at offset {msg.offset()}: {e}")
            self._send_to_dlq(msg, str(e))
            # Commit offset even on failure to avoid poison pills blocking the pipeline
            self.consumer.commit(asynchronous=False)

    def _predict_doom(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock inference function."""
        # Extract text or content from input
        content = data.get("text", "")
        source_id = data.get("user_id", str(uuid.uuid4()))

        # Mock logic
        doom_score = 0.85 if "doom" in content.lower() else 0.15

        if doom_score >= 0.8:
            risk_level = "RISK_LEVEL_EXTREME"
        elif doom_score >= 0.6:
            risk_level = "RISK_LEVEL_HIGH"
        elif doom_score >= 0.4:
            risk_level = "RISK_LEVEL_MEDIUM"
        else:
            risk_level = "RISK_LEVEL_LOW"

        return {
            "prediction_id": str(uuid.uuid4()),
            "source_id": source_id,
            "doom_score": doom_score,
            "risk_level": risk_level,
            "timestamp": int(time.time() * 1000),
        }

    def _publish_result(self, result: Dict[str, Any]) -> None:
        """Publish inference result to the output topic."""

        def delivery_callback(err, msg):
            if err:
                logger.error(f"Failed to deliver result to {self.output_topic}: {err}")
            else:
                logger.debug(f"Delivered result to {msg.topic()} [{msg.partition()}]")

        self.producer.produce(
            self.output_topic,
            key=result.get("source_id", "").encode("utf-8"),
            value=json.dumps(result).encode("utf-8"),
            callback=delivery_callback,
        )
        self.producer.poll(0)

    def _send_to_dlq(self, msg: Message, error_reason: str) -> None:
        """Send a failed message to the Dead Letter Queue."""
        logger.info(f"Sending message to DLQ: {self.dlq_topic}")

        dlq_payload = {
            "original_topic": msg.topic(),
            "original_partition": msg.partition(),
            "original_offset": msg.offset(),
            "error": error_reason,
            "raw_value": msg.value().decode("utf-8", errors="replace") if msg.value() else None,
            "timestamp": int(time.time() * 1000),
        }

        def dlq_delivery_callback(err, delivered_msg):
            if err:
                logger.critical(f"Failed to deliver message to DLQ: {err}")

        self.producer.produce(
            self.dlq_topic,
            key=msg.key(),
            value=json.dumps(dlq_payload).encode("utf-8"),
            callback=dlq_delivery_callback,
        )
        self.producer.poll(0)

    def _shutdown(self) -> None:
        """Clean shutdown of Kafka clients."""
        logger.info("Shutting down worker...")

        # Flush producer queue
        remaining = self.producer.flush(timeout=10.0)
        if remaining > 0:
            logger.warning(f"{remaining} messages failed to flush to broker")

        # Close consumer
        self.consumer.close()
        logger.info("Worker shutdown complete")


def main():
    # Load config from env or use defaults
    bootstrap_servers = "localhost:9092"
    group_id = "doom-inference-group"

    worker = DoomInferenceWorker(bootstrap_servers=bootstrap_servers, group_id=group_id)

    # Handle graceful shutdown
    def handle_signal(sig, frame):
        worker.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info(f"Starting worker connected to {bootstrap_servers}")
    worker.start()


if __name__ == "__main__":
    main()
