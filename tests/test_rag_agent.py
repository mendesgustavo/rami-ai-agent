from app.rag_agent import (
    AgentAnswer,
    create_system_prompt,
    extract_text_from_response,
    format_context,
    format_source_name,
)
from app.vector_store import SearchResult


def create_search_result() -> SearchResult:
    return SearchResult(
        chunk_id="faq-page-1-chunk-1",
        document_name="faq.pdf",
        page_number=1,
        content="A RAMI funciona exclusivamente como loja online.",
        distance=0.1,
    )


def test_extract_string_response() -> None:
    content = "Resposta de teste."

    assert extract_text_from_response(content) == content


def test_extract_structured_response() -> None:
    content = [
        {
            "type": "text",
            "text": "Resposta estruturada.",
        }
    ]

    assert (
        extract_text_from_response(content)
        == "Resposta estruturada."
    )


def test_format_context() -> None:
    result = create_search_result()

    context = format_context([result])

    assert "[TRECHO 1]" in context
    assert "faq.pdf" in context
    assert "Página: 1" in context
    assert result.content in context


def test_system_prompt_contains_safety_rules() -> None:
    prompt = create_system_prompt()

    assert "exclusivamente o contexto" in prompt
    assert "Não invente" in prompt
    assert "português do Brasil" in prompt


def test_format_source_name() -> None:
    name = format_source_name(
        "politica_de_privacidade.pdf"
    )

    assert name == "Politica De Privacidade"


def test_agent_answer_dataclass() -> None:
    result = create_search_result()

    answer = AgentAnswer(
        question="Pergunta",
        answer="Resposta",
        sources=[result],
    )

    assert answer.question == "Pergunta"
    assert answer.answer == "Resposta"
    assert len(answer.sources) == 1