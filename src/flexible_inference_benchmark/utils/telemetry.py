import os
from typing import Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, get_aggregated_resources
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor

DEFAULT_SERVICE_NAME = "centml-fib"


def setup_telemetry() -> None:
    """
    Set up OpenTelemetry tracing with OTLP exporter.
    The service name is determined in the following order:
    1. OTEL_SERVICE_NAME environment variable
    2. DEFAULT_SERVICE_NAME
    """
    # Check if telemetry is enabled
    if os.getenv("OTEL_ENABLED", "false").lower() != "true":
        return

    # Get service name from environment or use default
    service_name = os.getenv("OTEL_SERVICE_NAME", DEFAULT_SERVICE_NAME)

    # Create resource with service name
    resource = Resource.create({SERVICE_NAME: service_name})

    # Set up the tracer provider
    provider = TracerProvider(resource=resource)

    # Set up the OTLP exporter
    otlp_exporter = OTLPSpanExporter()

    # Add the span processor to the tracer provider
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter, max_queue_size=10000))

    # Set the tracer provider
    trace.set_tracer_provider(provider)

    # Instrument aiohttp client
    AioHttpClientInstrumentor().instrument()


def create_span_attributes(
    prompt_tokens: int, image_count: int, image_sizes: list[int], response_tokens: int, run_id: str
) -> dict[str, int | list[int] | str]:
    """
    Create a dictionary of span attributes for a request.
    All attributes are prefixed with 'fib.' for easy identification.
    Token counts are used instead of character sizes for better LLM metrics.
    """
    return {
        "fib.prompt.tokens": prompt_tokens,
        "fib.image.count": image_count,
        "fib.image.sizes": image_sizes,
        "fib.response.tokens": response_tokens,
        "fib.run.id": run_id,
    }
