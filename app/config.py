import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


def load_environment() -> None:
    """
    Carrega as variáveis definidas no arquivo .env.
    """
    load_dotenv(dotenv_path=ENV_FILE)


def get_google_api_key() -> str:
    """
    Obtém e valida a chave da API do Gemini.
    """
    load_environment()

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "A variável GOOGLE_API_KEY não foi configurada. "
            "Crie o arquivo .env com sua chave do Gemini."
        )

    return api_key


def get_gemini_model_name() -> str:
    """
    Obtém o modelo Gemini configurado no ambiente.
    """
    load_environment()

    model_name = os.getenv(
        "GEMINI_MODEL",
        DEFAULT_GEMINI_MODEL,
    ).strip()

    if not model_name:
        return DEFAULT_GEMINI_MODEL

    return model_name