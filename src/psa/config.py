from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "dev"
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://postgres:postgres@db:5432/sensor_analytics"

    aws_region: str = "us-east-1"
    s3_bucket_name: str | None = None
    sagemaker_endpoint_name: str | None = None


settings = Settings()
