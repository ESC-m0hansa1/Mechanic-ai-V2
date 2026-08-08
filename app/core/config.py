from pydantic_settings import BaseSettings, SettingsConfigDict


# A class that inherits from BaseSettings automatically reads matching
# environment variables (and the .env file) into typed attributes.
class Settings(BaseSettings):
    # model_config tells pydantic WHERE to read from and how to behave.
    # env_file=".env" -> load that file. extra="ignore" -> don't crash
    # if .env contains keys we haven't declared here yet.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Each attribute maps to an env var of the SAME NAME (case-insensitive).
    # The type annotation IS the validation: APP_PORT must parse as int.
    app_name: str          # no default -> REQUIRED. Missing => startup error.
    app_port: int = 8000   # default -> optional. Used if env var absent.


# Create ONE shared instance the whole app imports. Built once at startup,
# so the .env file is read a single time, not on every request.
settings = Settings()