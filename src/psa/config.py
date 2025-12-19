from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "dev"
    log_level: str = "INFO"

    enable_db: bool = True

    database_url: str = "postgresql+asyncpg://postgres:postgres@db:5432/sensor_analytics"

    model_family: str = "iforest"  # iforest|autoencoder|lstm
    model_artifact_path: str | None = None

    feature_cols: str = "pressure,force,acceleration,temperature"

    aws_region: str = "us-east-1"
    s3_bucket_name: str | None = None
    sagemaker_endpoint_name: str | None = None


settings = Settings()
