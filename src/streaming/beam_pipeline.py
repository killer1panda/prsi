"""
Apache Beam pipeline for large-scale ETL on social media data.
Supports both batch (historical backfill) and streaming (real-time) modes.
"""
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions, SetupOptions
from apache_beam.transforms.window import FixedWindows, SlidingWindows
from apache_beam.transforms.trigger import AfterWatermark, AfterProcessingTime, AccumulationMode

logger = logging.getLogger(__name__)


@dataclass
class BeamConfig:
    runner: str = "DirectRunner"  # or "DataflowRunner"
    project: str = "doom-index"
    region: str = "us-central1"
    temp_location: str = "gs://doom-index-bucket/temp"
    staging_location: str = "gs://doom-index-bucket/staging"
    streaming: bool = False
    window_size: int = 60  # seconds
    num_workers: int = 4
    max_num_workers: int = 20
    autoscaling_algorithm: str = "THROUGHPUT_BASED"


class ExtractPostFn(beam.DoFn):
    """Parse raw JSON post into structured dict."""

    def process(self, element: str):
        import json
        try:
            post = json.loads(element)
            yield {
                "post_id": post.get("id", ""),
                "user_id": post.get("author", ""),
                "text": post.get("text", ""),
                "timestamp": post.get("created_utc", 0),
                "subreddit": post.get("subreddit", ""),
                "score": post.get("score", 0),
                "num_comments": post.get("num_comments", 0),
                "source": post.get("source", "unknown")
            }
        except json.JSONDecodeError:
            yield beam.pvalue.TaggedOutput("parse_errors", element)


class EnrichFeaturesFn(beam.DoFn):
    """Enrich posts with ML features."""

    def __init__(self, feature_extractor: Callable):
        self.feature_extractor = feature_extractor
        self._model = None

    def setup(self):
        # Lazy load model (serialized via pickle in Beam)
        pass

    def process(self, post: Dict):
        try:
            features = self.feature_extractor(post)
            post["features"] = features
            yield post
        except Exception as e:
            post["error"] = str(e)
            yield beam.pvalue.TaggedOutput("enrichment_errors", post)


class PredictDoomFn(beam.DoFn):
    """Run Doom Index prediction on enriched posts."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None

    def setup(self):
        # Load model once per worker
        import torch
        self._model = torch.load(self.model_path, map_location="cpu")
        self._model.eval()

    def process(self, post: Dict):
        try:
            # Run inference
            features = post.get("features", {})
            # Convert to model input and predict
            # This is a placeholder - actual implementation depends on model format
            doom_score = 0.5  # Placeholder

            yield {
                "post_id": post["post_id"],
                "user_id": post["user_id"],
                "doom_score": doom_score,
                "timestamp": post["timestamp"],
                "features": features
            }
        except Exception as e:
            post["error"] = str(e)
            yield beam.pvalue.TaggedOutput("prediction_errors", post)


class FormatOutputFn(beam.DoFn):
    """Format output for BigQuery/Parquet."""

    def process(self, prediction: Dict):
        yield {
            "post_id": prediction["post_id"],
            "user_id": prediction["user_id"],
            "doom_score": float(prediction["doom_score"]),
            "timestamp": prediction["timestamp"],
            "feature_json": str(prediction.get("features", {}))
        }


class DoomBeamPipeline:
    """
    Apache Beam pipeline for Doom Index ETL.
    """

    def __init__(self, config: Optional[BeamConfig] = None):
        self.config = config or BeamConfig()
        logger.info(f"BeamPipeline initialized: runner={self.config.runner}")

    def _get_options(self) -> PipelineOptions:
        """Build pipeline options."""
        options = PipelineOptions()

        if self.config.runner == "DataflowRunner":
            options.view_as(StandardOptions).runner = "DataflowRunner"
            options.view_as(SetupOptions).requirements_file = "requirements.txt"
        else:
            options.view_as(StandardOptions).runner = "DirectRunner"

        if self.config.streaming:
            options.view_as(StandardOptions).streaming = True

        return options

    def build_batch_pipeline(self, input_path: str, output_path: str,
                             feature_extractor: Callable, model_path: str):
        """
        Build batch pipeline for historical data processing.

        Args:
            input_path: GCS path to raw NDJSON files
            output_path: GCS path for output Parquet
            feature_extractor: Function to extract features from post dict
            model_path: Path to serialized model
        """
        options = self._get_options()

        with beam.Pipeline(options=options) as p:
            raw = p | "ReadRaw" >> beam.io.ReadFromText(input_path)

            parsed = raw | "Parse" >> beam.ParDo(ExtractPostFn())

            enriched = parsed | "Enrich" >> beam.ParDo(
                EnrichFeaturesFn(feature_extractor)
            )

            predictions = enriched | "Predict" >> beam.ParDo(
                PredictDoomFn(model_path)
            )

            formatted = predictions | "Format" >> beam.ParDo(FormatOutputFn())

            # Write to Parquet
            formatted | "WriteParquet" >> beam.io.WriteToParquet(
                file_path_prefix=output_path,
                schema=self._get_parquet_schema(),
                file_name_suffix=".parquet"
            )

    def build_streaming_pipeline(self, input_subscription: str, 
                                 output_table: str,
                                 feature_extractor: Callable, 
                                 model_path: str):
        """
        Build streaming pipeline from Pub/Sub to BigQuery.

        Args:
            input_subscription: Pub/Sub subscription path
            output_table: BigQuery table spec (project:dataset.table)
            feature_extractor: Feature extraction function
            model_path: Model path
        """
        options = self._get_options()
        options.view_as(StandardOptions).streaming = True

        with beam.Pipeline(options=options) as p:
            raw = p | "ReadPubSub" >> beam.io.ReadFromPubSub(
                subscription=input_subscription
            )

            # Windowing
            windowed = raw | "Window" >> beam.WindowInto(
                FixedWindows(size=self.config.window_size),
                trigger=AfterWatermark(
                    early=AfterProcessingTime(30),
                    late=AfterProcessingTime(60)
                ),
                allowed_lateness=300,
                accumulation_mode=AccumulationMode.ACCUMULATING
            )

            parsed = windowed | "Parse" >> beam.ParDo(ExtractPostFn())
            enriched = parsed | "Enrich" >> beam.ParDo(EnrichFeaturesFn(feature_extractor))
            predictions = enriched | "Predict" >> beam.ParDo(PredictDoomFn(model_path))
            formatted = predictions | "Format" >> beam.ParDo(FormatOutputFn())

            formatted | "WriteBQ" >> beam.io.WriteToBigQuery(
                table=output_table,
                schema=self._get_bq_schema(),
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
            )

    def _get_parquet_schema(self):
        """Get PyArrow schema for Parquet output."""
        import pyarrow as pa
        return pa.schema([
            ("post_id", pa.string()),
            ("user_id", pa.string()),
            ("doom_score", pa.float64()),
            ("timestamp", pa.int64()),
            ("feature_json", pa.string())
        ])

    def _get_bq_schema(self):
        """Get BigQuery schema."""
        from apache_beam.io.gcp.bigquery import TableFieldSchema
        return {
            "fields": [
                {"name": "post_id", "type": "STRING", "mode": "REQUIRED"},
                {"name": "user_id", "type": "STRING", "mode": "REQUIRED"},
                {"name": "doom_score", "type": "FLOAT", "mode": "NULLABLE"},
                {"name": "timestamp", "type": "TIMESTAMP", "mode": "NULLABLE"},
                {"name": "feature_json", "type": "STRING", "mode": "NULLABLE"}
            ]
        }
