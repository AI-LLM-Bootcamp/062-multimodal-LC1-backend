import boto3
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

    # Database settings
    DATABASE_HOST: str
    DATABASE_NAME: str
    DATABASE_USER: str
    DATABASE_PASSWORD: str
    DATABASE_PORT: int

    # App settings
    app_name: str = "Full Stack Multimodal PDF App with LC1"

    # AWS settings
    AWS_KEY: str
    AWS_SECRET: str

    # Unstructured API settings
    UNSTRUCTURED_API_KEY: str
    AWS_S3_BUCKET: str = "pdf-basic-app-julio-012726"

    def get_s3_client(self):
        """Create and return an S3 client instance"""
        return boto3.client(
            's3',
            aws_access_key_id=self.AWS_KEY,
            aws_secret_access_key=self.AWS_SECRET
        )

@lru_cache
def get_settings() -> Settings:
    """
    Create and cache a Settings instance.
    This ensures settings are loaded only once.
    """
    return Settings()
