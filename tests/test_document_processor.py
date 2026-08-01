import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.document_loader import DocumentPage, load_all_documents
from app.document_processor import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    create_chunks_from_page,
    create_text_splitter,
    process_all_documents,
    process_document_pages,
)


def test_create_text_splitter() -> None:
    splitter = create_text_splitter()

    assert isinstance(
        splitter,
        RecursiveCharacterTextSplitter,
    )


def test_invalid_chunk_size() -> None:
    with pytest.raises(ValueError):
        create_text_splitter(chunk_size=0)


def test_invalid_negative_overlap() -> None:
    with pytest.raises(ValueError):
        create_text_splitter(chunk_overlap=-1)


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        create_text_splitter(
            chunk_size=500,
            chunk_overlap=500,
        )


def test_create_chunks_from_page() -> None:
    page = DocumentPage(
        document_name="documento_teste.pdf",
        page_number=3,
        content=(
            "Este é um texto de teste. " * 100
        ),
    )

    splitter = create_text_splitter(
        chunk_size=200,
        chunk_overlap=40,
    )

    chunks = create_chunks_from_page(page, splitter)

    assert len(chunks) > 1
    assert all(
        chunk.document_name == "documento_teste.pdf"
        for chunk in chunks
    )
    assert all(
        chunk.page_number == 3
        for chunk in chunks
    )
    assert chunks[0].chunk_id == (
        "documento_teste-page-3-chunk-1"
    )


def test_process_document_pages() -> None:
    pages = load_all_documents()
    chunks = process_document_pages(pages)

    assert len(chunks) > len(pages)
    assert all(chunk.content.strip() for chunk in chunks)
    assert all(chunk.chunk_id for chunk in chunks)
    assert all(chunk.page_number > 0 for chunk in chunks)


def test_default_chunk_configuration() -> None:
    assert DEFAULT_CHUNK_SIZE == 1000
    assert DEFAULT_CHUNK_OVERLAP == 200


def test_process_all_documents() -> None:
    chunks = process_all_documents()

    document_names = {
        chunk.document_name
        for chunk in chunks
    }

    assert len(document_names) == 5
    assert len(chunks) > 0