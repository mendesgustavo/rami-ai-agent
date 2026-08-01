from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.document_loader import DocumentPage, load_all_documents


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


@dataclass
class DocumentChunk:
    """
    Representa um trecho de texto extraído da base de conhecimento.
    """

    chunk_id: str
    document_name: str
    page_number: int
    chunk_index: int
    content: str


def create_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """
    Cria o separador de texto utilizado pelo projeto.
    """
    if chunk_size <= 0:
        raise ValueError("O tamanho do chunk deve ser maior que zero.")

    if chunk_overlap < 0:
        raise ValueError("A sobreposição não pode ser negativa.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "A sobreposição deve ser menor que o tamanho do chunk."
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )


def create_chunks_from_page(
    page: DocumentPage,
    splitter: RecursiveCharacterTextSplitter,
) -> list[DocumentChunk]:
    """
    Divide uma página em trechos menores e preserva seus metadados.
    """
    texts = splitter.split_text(page.content)
    chunks: list[DocumentChunk] = []

    document_identifier = page.document_name.removesuffix(".pdf")

    for chunk_index, text in enumerate(texts, start=1):
        normalized_text = text.strip()

        if not normalized_text:
            continue

        chunk_id = (
            f"{document_identifier}"
            f"-page-{page.page_number}"
            f"-chunk-{chunk_index}"
        )

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_name=page.document_name,
                page_number=page.page_number,
                chunk_index=chunk_index,
                content=normalized_text,
            )
        )

    return chunks


def process_document_pages(
    pages: list[DocumentPage],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """
    Divide todas as páginas recebidas em chunks.
    """
    splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks: list[DocumentChunk] = []

    for page in pages:
        page_chunks = create_chunks_from_page(page, splitter)
        all_chunks.extend(page_chunks)

    return all_chunks


def process_all_documents(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """
    Carrega os PDFs e processa toda a base de conhecimento.
    """
    pages = load_all_documents()

    return process_document_pages(
        pages=pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def print_processing_summary(
    pages: list[DocumentPage],
    chunks: list[DocumentChunk],
) -> None:
    """
    Exibe um resumo do processamento dos documentos.
    """
    document_names = sorted(
        {chunk.document_name for chunk in chunks}
    )

    total_characters = sum(
        len(chunk.content) for chunk in chunks
    )

    print("Resumo do processamento")
    print("-----------------------")
    print(f"Documentos processados: {len(document_names)}")
    print(f"Páginas processadas: {len(pages)}")
    print(f"Chunks gerados: {len(chunks)}")
    print(f"Caracteres nos chunks: {total_characters}")
    print()

    for document_name in document_names:
        document_chunks = [
            chunk
            for chunk in chunks
            if chunk.document_name == document_name
        ]

        print(
            f"- {document_name}: "
            f"{len(document_chunks)} chunks"
        )


def main() -> None:
    """
    Executa uma validação manual do processamento.
    """
    pages = load_all_documents()
    chunks = process_document_pages(pages)

    print_processing_summary(pages, chunks)

    if not chunks:
        raise RuntimeError(
            "Nenhum chunk foi criado a partir dos documentos."
        )

    first_chunk = chunks[0]

    print("\nAmostra do primeiro chunk")
    print("-------------------------")
    print(f"ID: {first_chunk.chunk_id}")
    print(f"Documento: {first_chunk.document_name}")
    print(f"Página: {first_chunk.page_number}")
    print(f"Índice na página: {first_chunk.chunk_index}")
    print(f"Quantidade de caracteres: {len(first_chunk.content)}")
    print()
    print(first_chunk.content)


if __name__ == "__main__":
    main()