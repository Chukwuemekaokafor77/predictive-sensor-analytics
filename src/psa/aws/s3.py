from __future__ import annotations

from typing import BinaryIO

import boto3

from psa.config import settings


def get_s3_client():
    return boto3.client("s3", region_name=settings.aws_region)


def upload_fileobj(*, fileobj: BinaryIO, key: str) -> None:
    if not settings.s3_bucket_name:
        raise RuntimeError("S3_BUCKET_NAME is not configured")
    s3 = get_s3_client()
    s3.upload_fileobj(fileobj, settings.s3_bucket_name, key)
