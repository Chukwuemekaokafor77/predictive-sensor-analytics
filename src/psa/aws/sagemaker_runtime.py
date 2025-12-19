from __future__ import annotations

import json
from typing import Any

import boto3

from psa.config import settings


def get_sm_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=settings.aws_region)


def invoke_endpoint(*, payload: dict[str, Any]) -> dict[str, Any]:
    if not settings.sagemaker_endpoint_name:
        raise RuntimeError("SAGEMAKER_ENDPOINT_NAME is not configured")

    sm = get_sm_runtime_client()
    resp = sm.invoke_endpoint(
        EndpointName=settings.sagemaker_endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )
    body = resp["Body"].read().decode("utf-8")
    return json.loads(body)
