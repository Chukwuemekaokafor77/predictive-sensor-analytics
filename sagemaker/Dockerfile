FROM python:3.11-slim

WORKDIR /opt/ml/code

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY sagemaker/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Code
COPY src /opt/ml/code/src
COPY sagemaker /opt/ml/code/sagemaker

ENV PYTHONPATH=/opt/ml/code/src

# SageMaker convention: inference container listens on 8080
EXPOSE 8080

# Default to inference; training uses `python sagemaker/train.py`
CMD ["python", "sagemaker/serve.py"]
