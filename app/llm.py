from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import (
    get_gemini_model_name,
    get_google_api_key,
)


def create_llm() -> ChatGoogleGenerativeAI:
    """
    Cria o modelo Gemini utilizado pelo agente.
    """
    api_key = get_google_api_key()
    model_name = get_gemini_model_name()

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        max_retries=2,
        timeout=60,
    )


def test_llm_connection() -> str:
    """
    Realiza uma chamada simples para validar a integração.
    """
    llm = create_llm()

    response = llm.invoke(
        [
            (
                "system",
                (
                    "Você é um assistente de teste. "
                    "Responda sempre em português do Brasil."
                ),
            ),
            (
                "human",
                (
                    "Responda somente com a frase: "
                    "Conexão com o Gemini realizada com sucesso."
                ),
            ),
        ]
    )

    content = response.content

    if not content:
        raise RuntimeError(
            "O Gemini retornou uma resposta vazia."
        )

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []

        for block in content:
            if isinstance(block, dict):
                text = block.get("text")

                if text:
                    text_parts.append(str(text).strip())

        if text_parts:
            return "\n".join(text_parts)

    return str(content).strip()


def main() -> None:
    """
    Testa a comunicação entre o projeto e o Gemini.
    """
    model_name = get_gemini_model_name()

    print("Testando conexão com o Gemini...")
    print(f"Modelo configurado: {model_name}")
    print()

    answer = test_llm_connection()

    print("Resposta recebida:")
    print(answer)


if __name__ == "__main__":
    main()