#!/usr/bin/env python3
"""
Invoke a Fission function, inject a W3C traceparent header, then query Jaeger by trace_id
and report the end-to-end latency (and router span latency if present).

Prereqs:
  pip install requests opentelemetry-sdk

Environment variables (recommended):
  FISSION_URL   - full function URL (e.g., http://<router>/v2/functions/<name>)
  JAEGER_URL    - Jaeger query base URL (e.g., http://jaeger-query.observability.svc:16686)
Optional:
  METHOD        - HTTP method (default POST)
  PAYLOAD       - JSON payload string (default {"ping":"pong"})
  LOOKBACK_S    - how long to poll Jaeger (default 30)
  POLL_S        - poll interval seconds (default 1)
"""

import os
import time
import json
import sys
import requests

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry import trace


def make_traceparent() -> tuple[str, str]:
    """
    Create a fresh trace/span locally and format a W3C traceparent header.
    Returns (trace_id_hex, traceparent_header_value).
    """
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer("client")

    with tracer.start_as_current_span("client.invoke") as span:
        sc = span.get_span_context()
        trace_id_hex = format(sc.trace_id, "032x")
        span_id_hex = format(sc.span_id, "016x")
        # version 00, flags 01 (sampled). Adjust flags if you want.
        traceparent = f"00-{trace_id_hex}-{span_id_hex}-01"
        return trace_id_hex, traceparent


def invoke_function(url: str, method: str, payload: dict, traceparent: str) -> requests.Response:
    headers = {
        "traceparent": traceparent,
        # "content-type": "application/json",
        # "accept": "application/json",
    }
    if method.upper() == "POST":
        return requests.post(url, headers=headers, json=payload, timeout=30)
    if method.upper() == "GET":
        return requests.get(url, headers=headers, timeout=30)
    if method.upper() == "PUT":
        return requests.put(url, headers=headers, json=payload, timeout=30)
    if method.upper() == "DELETE":
        return requests.delete(url, headers=headers, timeout=30)
    raise ValueError(f"Unsupported METHOD={method}")


def jaeger_get_trace(jaeger_base: str, trace_id_hex: str) -> dict | None:
    """
    Query Jaeger by trace ID. Returns trace JSON dict if found, else None.
    """
    url = f"{jaeger_base.rstrip('/')}/api/traces/{trace_id_hex}"
    r = requests.get(url, timeout=15)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def compute_trace_latency_us(jaeger_trace_json: dict) -> tuple[int, dict]:
    """
    Jaeger /api/traces/<id> returns:
      { "data": [ { "spans": [...], ... } ] }

    We compute:
      - overall trace latency = (max end_time - min start_time) across all spans
      - also attempt to find the Fission-Router span latency, if present
    Returns (overall_latency_us, details_dict)
    """
    data = jaeger_trace_json.get("data") or []
    if not data:
        raise ValueError("Jaeger returned no trace data.")

    trace_obj = data[0]
    spans = trace_obj.get("spans") or []

    if not spans:
        raise ValueError("Trace contains no spans.")

    # Jaeger times: startTime in microseconds since epoch, duration in microseconds
    min_start = min(s["startTime"] for s in spans)
    max_end = max(s["startTime"] + s["duration"] for s in spans)
    overall_us = max_end - min_start

    # Best-effort: locate router span
    router_span = None
    for s in spans:
        # Process/service name is stored in trace_obj["processes"][span["processID"]]["serviceName"]
        # We'll resolve it below and match "Fission-Router" as in your screenshot.
        pass

    processes = trace_obj.get("processes") or {}
    # Resolve serviceName for each span
    enriched = []
    for s in spans:
        proc = processes.get(s.get("processID"), {})
        svc = proc.get("serviceName")
        enriched.append((svc, s))

    # Try common matches
    for svc, s in enriched:
        if svc and svc.lower() in {"fission-router", "fissionrouter", "router"}:
            router_span = s
            break

    details = {
        "span_count": len(spans),
        "router_span_duration_us": router_span["duration"] if router_span else None,
        "router_span_operation": router_span.get("operationName") if router_span else None,
    }
    return overall_us, details


def main():
    function_name = "ml-image-processing"
    fission_url = f"http://localhost:31314/{function_name}"
    jaeger_url = "http://localhost:32391"
    method = "POST"

    if not fission_url or not jaeger_url:
        print("ERROR: Set FISSION_URL and JAEGER_URL environment variables.", file=sys.stderr)
        print("Example:", file=sys.stderr)
        print('  export FISSION_URL="http://<router>/v2/functions/<fn>"', file=sys.stderr)
        print('  export JAEGER_URL="http://jaeger-query.<ns>.svc:16686"', file=sys.stderr)
        sys.exit(2)

    payload = {"image_name": "000b7b74-0a22-4d0c-b717-e240fdc5d555.png"}
    lookback_s = 5
    poll_s = 0.5

    trace_id, traceparent = make_traceparent()
    print(f"trace_id={trace_id}")
    print(f"traceparent={traceparent}")

    # Invoke
    t0 = time.time()
    resp = invoke_function(fission_url, method, payload, traceparent)
    t1 = time.time()

    print(f"invoke_http_status={resp.status_code}")
    print(f"client_observed_rtt_ms={(t1 - t0) * 1000:.2f}")
    # Print small response body if possible (avoid huge output)
    body_preview = resp.text
    if len(body_preview) > 400:
        body_preview = body_preview[:400] + "...(truncated)"
    print(f"response_preview={body_preview}")

    # Poll Jaeger until trace arrives
    deadline = time.time() + lookback_s
    last_err = None
    while time.time() < deadline:
        try:
            jt = jaeger_get_trace(jaeger_url, trace_id)
            if jt:
                overall_us, details = compute_trace_latency_us(jt)
                print(f"jaeger_trace_found=true")
                print(f"trace_latency_ms={overall_us / 1000.0:.2f}")
                if details["router_span_duration_us"] is not None:
                    print(f"router_span_operation={details['router_span_operation']}")
                    print(f"router_span_latency_ms={details['router_span_duration_us'] / 1000.0:.2f}")
                else:
                    print("router_span_latency_ms=NA (router span not identified; trace still valid)")
                print(f"span_count={details['span_count']}")
                return
        except Exception as e:
            last_err = e

        time.sleep(poll_s)

    print("jaeger_trace_found=false", file=sys.stderr)
    if last_err:
        print(f"last_error={last_err}", file=sys.stderr)
    print(
        "Notes: if Jaeger never finds the trace, verify (1) Fission router is propagating W3C tracecontext,"
        " (2) Jaeger Query URL is correct, (3) sampling is 100% on the router side.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
