import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def setup_telemetry(app, service_name="doom-backend"):
    """
    Configure OpenTelemetry TracerProvider, OTLPSpanExporter,
    and automatically instrument FastAPI and Requests.
    """
    resource = Resource(attributes={"service.name": service_name})

    provider = TracerProvider(resource=resource)

    # Configure OTLP Exporter (defaults to localhost:4317 for Jaeger OTLP receiver)
    otlp_endpoint = os.environ.get("OTLP_ENDPOINT", "http://localhost:4317")
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)

    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)

    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)

    # Instrument Requests
    RequestsInstrumentor().instrument()

    return provider
