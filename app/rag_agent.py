from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.llm import create_llm
from app.vector_store import SearchResult, search_similar_chunks


DEFAULT_RETRIEVAL_LIMIT = 5


@dataclass
class AgentAnswer:
    """
    Representa a resposta final produzida pelo agente.
    """

    question: str
    answer: str
    sources: list[SearchResult]


def extract_text_from_response(content: object) -> str:
    """
    Extrai texto de respostas simples ou estruturadas do Gemini.
    """
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


def format_context(results: list[SearchResult]) -> str:
    """
    Organiza os trechos recuperados em um contexto para o modelo.
    """
    if not results:
        return "Nenhum trecho relevante foi encontrado."

    context_parts: list[str] = []

    for position, result in enumerate(results, start=1):
        context_parts.append(
            "\n".join(
                [
                    f"[TRECHO {position}]",
                    f"Documento: {result.document_name}",
                    f"Página: {result.page_number}",
                    f"Conteúdo:",
                    result.content,
                ]
            )
        )

    return "\n\n".join(context_parts)


def create_system_prompt() -> str:
    """
    Define as regras de comportamento do agente.
    """
    return """
Você é o assistente virtual da RAMI, uma loja online de roupas femininas.

Sua função é responder perguntas usando exclusivamente o contexto fornecido
a partir dos documentos internos da RAMI.

Siga rigorosamente estas regras:

1. Responda sempre em português do Brasil.
2. Utilize somente informações presentes no contexto.
3. Não invente políticas, prazos, preços, contatos ou procedimentos.
4. Quando a resposta não estiver no contexto, diga claramente:
   "Não encontrei essa informação nos documentos da RAMI."
5. Responda de forma clara, direta, educada e natural.
6. Quando houver prazos, destaque se são dias úteis ou dias corridos.
7. Não mencione conceitos técnicos como embeddings, chunks, RAG ou banco vetorial.
8. Não diga que recebeu trechos de contexto.
9. Não crie links ou contatos que não estejam nos documentos.
10. Em caso de informações complementares, organize a resposta em parágrafos
    curtos ou tópicos.
""".strip()


def create_user_prompt(
    question: str,
    context: str,
) -> str:
    """
    Monta a mensagem contendo o contexto e a pergunta.
    """
    return f"""
Use os documentos abaixo para responder à pergunta.

DOCUMENTOS RECUPERADOS:

{context}

PERGUNTA DA CLIENTE:

{question}

Produza uma resposta final objetiva e completa.
Não inclua uma seção de fontes, pois as fontes serão exibidas separadamente
pela aplicação.
""".strip()


def generate_answer(
    question: str,
    results: list[SearchResult],
    llm: ChatGoogleGenerativeAI,
) -> str:
    """
    Envia contexto e pergunta ao Gemini.
    """
    context = format_context(results)

    response = llm.invoke(
        [
            SystemMessage(
                content=create_system_prompt(),
            ),
            HumanMessage(
                content=create_user_prompt(
                    question=question,
                    context=context,
                ),
            ),
        ]
    )

    answer = extract_text_from_response(response.content)

    if not answer:
        raise RuntimeError(
            "O Gemini retornou uma resposta vazia."
        )

    return answer


def ask_agent(
    question: str,
    retrieval_limit: int = DEFAULT_RETRIEVAL_LIMIT,
    llm: ChatGoogleGenerativeAI | None = None,
) -> AgentAnswer:
    """
    Executa o fluxo completo de recuperação e geração.
    """
    normalized_question = question.strip()

    if not normalized_question:
        raise ValueError(
            "A pergunta não pode estar vazia."
        )

    if retrieval_limit <= 0:
        raise ValueError(
            "A quantidade de resultados deve ser maior que zero."
        )

    results = search_similar_chunks(
        question=normalized_question,
        result_limit=retrieval_limit,
    )

    if llm is None:
        llm = create_llm()

    answer = generate_answer(
        question=normalized_question,
        results=results,
        llm=llm,
    )

    return AgentAnswer(
        question=normalized_question,
        answer=answer,
        sources=results,
    )


def format_source_name(document_name: str) -> str:
    """
    Converte o nome técnico do arquivo em um título legível.
    """
    name_without_extension = document_name.removesuffix(".pdf")

    return name_without_extension.replace("_", " ").title()


def print_agent_answer(agent_answer: AgentAnswer) -> None:
    """
    Exibe a resposta e as fontes no terminal.
    """
    print()
    print("=" * 70)
    print("RESPOSTA DO AGENTE RAMI")
    print("=" * 70)
    print()
    print(agent_answer.answer)

    print()
    print("Fontes consultadas")
    print("------------------")

    displayed_sources: set[tuple[str, int]] = set()

    for source in agent_answer.sources:
        source_key = (
            source.document_name,
            source.page_number,
        )

        if source_key in displayed_sources:
            continue

        displayed_sources.add(source_key)

        readable_name = format_source_name(
            source.document_name
        )

        print(
            f"- {readable_name}, página "
            f"{source.page_number}"
        )


def main() -> None:
    """
    Executa uma pergunta de teste no terminal.
    """
    question = (
        "Qual é o prazo para trocar uma roupa "
        "por outro tamanho?"
    )

    print("Pergunta enviada ao agente:")
    print(question)

    agent_answer = ask_agent(question)

    print_agent_answer(agent_answer)


if __name__ == "__main__":
    main()