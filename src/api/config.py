import os
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env file from the current directory
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

class Settings(BaseModel):
    PROJECT_NAME: str = "askGuru-SQL API"
    
    # Model Endpoints (vLLM)
    PRIMARY_MODEL_URL: str = os.getenv("PRIMARY_MODEL_URL", "http://localhost:8000/v1")
    SECONDARY_MODEL_URL: str = os.getenv("SECONDARY_MODEL_URL", "http://localhost:8001/v1")
    
    # Ensemble Control
    ENABLE_SECONDARY_MODEL: bool = os.getenv("ENABLE_SECONDARY_MODEL", "false").lower() == "true"
    ENSEMBLE_STRATEGY: str = os.getenv("ENSEMBLE_STRATEGY", "fallback") # "fallback" or "voting"
    
    # Database Settings (Oracle Thin Client)
    ORACLE_USER: str = os.getenv("DATABASE_USER", os.getenv("ORACLE_USER", "apps"))
    ORACLE_PASSWORD: str = os.getenv("DATABASE_PASSWORD", os.getenv("ORACLE_PASSWORD", "apps"))
    ORACLE_DSN: str = os.getenv("DATABASE_DSN", os.getenv("ORACLE_DSN", "183.82.4.173:1529/VIS"))
    ORACLE_LIB_DIR: str = os.getenv("ORACLE_CLIENT_LIB_UBUNTU", "")
    
    # M-Schema Path
    MSCHEMA_PATH: str = os.getenv("MSCHEMA_PATH", "data/data_warehouse/oracle_train/askGuru_m-schema_converted.txt")
    
    # Logic Settings
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "512"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))

settings = Settings()
