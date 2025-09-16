from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "EdgeServe-AB"
    env: str = "dev"
    port: int = 8080

    # Admin
    admin_token: str = "admin"

    # Observability
    prometheus_enabled: bool = True

    # Models
    model_va_path: str = "models/vA.onnx"
    model_vb_path: str = "models/vB.onnx"

    # Runtime & batching
    batch_max_size: int = 8
    batch_max_wait_ms: int = 2
    queue_max: int = 2048
    req_timeout_ms: int = 150
    ort_providers: str = "AUTO"

    # A/B & rollout
    ab_weight_a: float = 0.90
    ab_weight_b: float = 0.10
    canary_enabled: bool = True
    shadow_enabled: bool = True
    sticky_cookie: str = "ab_group"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Limits & resilience
    rate_limit_rps: int = 100  # tokens per second (per client)
    rate_limit_burst: int = 50  # bucket size
    max_body_bytes: int = 1_000_000  # ~1 MB
    cb_fail_threshold: int = 5  # consecutive failures to open circuit
    cb_reset_after_s: float = 30.0  # how long circuit stays open


settings = Settings()
