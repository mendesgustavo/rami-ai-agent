from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from app.document_processor import DocumentChunk, process_all_documents


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIRECTORY = PROJECT_ROOT / "vector_store"

COLLECTION_NAME = "rami_knowledge_base"

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

DEFAULT_RESULT_LIMIT = 5
DEFAULT_BATCH_SIZE = 32


@dataclass
class SearchResult:
    """
    Representa um trecho recuperado pela busca semântica.
    """

    chunk_id: str
    document_name: str
    page_number: int
    content: str
    distance: float


def create_embedding_model() -> SentenceTransformer:
    """
    Carrega o modelo multilíngue utilizado para gerar embeddings.
    """
    print(f"Carregando modelo de embeddings: {EMBEDDING_MODEL_NAME}")

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def create_chroma_client() -> chromadb.PersistentClient:
    """
    Cria um cliente persistente do ChromaDB.
    """
    VECTOR_STORE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    return chromadb.PersistentClient(
        path=str(VECTOR_STORE_DIRECTORY),
    )


def get_or_create_collection(
    client: chromadb.PersistentClient,
) -> Collection:
    """
    Obtém ou cria a coleção da base de conhecimento.
    """
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": (
                "Documentos internos da loja feminina RAMI"
            ),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "hnsw:space": "cosine",
        },
    )


def recreate_collection(
    client: chromadb.PersistentClient,
) -> Collection:
    """
    Remove a coleção anterior e cria uma nova coleção vazia.
    """
    try:
        client.delete_collection(
            name=COLLECTION_NAME,
        )

        print("Coleção anterior removida.")

    except Exception:
        print("Nenhuma coleção anterior foi encontrada.")

    return get_or_create_collection(client)


def generate_embeddings(
    texts: list[str],
    model: SentenceTransformer,
    text_type: str,
) -> list[list[float]]:
    """
    Gera embeddings normalizados para perguntas ou passagens.

    O modelo E5 exige:
    - "query: " para perguntas;
    - "passage: " para trechos dos documentos.
    """
    if not texts:
        return []

    if text_type not in {"query", "passage"}:
        raise ValueError(
            "O tipo do texto deve ser 'query' ou 'passage'."
        )

    prefixed_texts = [
        f"{text_type}: {text}"
        for text in texts
    ]

    embeddings = model.encode(
        prefixed_texts,
        batch_size=DEFAULT_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return embeddings.tolist()


def add_chunks_to_collection(
    chunks: list[DocumentChunk],
    collection: Collection,
    model: SentenceTransformer,
    batch_size: int = 100,
) -> None:
    """
    Adiciona os chunks e seus embeddings ao ChromaDB em lotes.
    """
    if not chunks:
        raise ValueError(
            "Nenhum chunk foi informado para indexação."
        )

    total_chunks = len(chunks)

    for start_index in range(0, total_chunks, batch_size):
        batch = chunks[
            start_index:start_index + batch_size
        ]

        texts = [
            chunk.content
            for chunk in batch
        ]

        embeddings = generate_embeddings(
            texts=texts,
            model=model,
            text_type="passage",
        )

        ids = [
            chunk.chunk_id
            for chunk in batch
        ]

        metadatas = [
            {
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in batch
        ]

        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        indexed_count = min(
            start_index + batch_size,
            total_chunks,
        )

        print(
            f"Chunks indexados: "
            f"{indexed_count}/{total_chunks}"
        )


def build_vector_store() -> int:
    """
    Processa os PDFs e reconstrói a base vetorial completa.
    """
    print("Processando documentos...")

    chunks = process_all_documents()

    if not chunks:
        raise RuntimeError(
            "Nenhum chunk foi gerado para indexação."
        )

    print(f"Total de chunks encontrados: {len(chunks)}")

    model = create_embedding_model()
    client = create_chroma_client()
    collection = recreate_collection(client)

    add_chunks_to_collection(
        chunks=chunks,
        collection=collection,
        model=model,
    )

    stored_count = collection.count()

    print()
    print("Base vetorial criada com sucesso.")
    print(f"Chunks armazenados: {stored_count}")
    print(f"Diretório: {VECTOR_STORE_DIRECTORY}")

    return stored_count


def search_similar_chunks(
    question: str,
    result_limit: int = DEFAULT_RESULT_LIMIT,
    model: SentenceTransformer | None = None,
    collection: Collection | None = None,
) -> list[SearchResult]:
    """
    Busca os chunks semanticamente mais próximos da pergunta.
    """
    normalized_question = question.strip()

    if not normalized_question:
        raise ValueError(
            "A pergunta não pode estar vazia."
        )

    if result_limit <= 0:
        raise ValueError(
            "A quantidade de resultados deve ser maior que zero."
        )

    if model is None:
        model = create_embedding_model()

    if collection is None:
        client = create_chroma_client()
        collection = get_or_create_collection(client)

    if collection.count() == 0:
        raise RuntimeError(
            "A base vetorial está vazia. "
            "Execute primeiro: py -m app.vector_store"
        )

    question_embedding = generate_embeddings(
        texts=[normalized_question],
        model=model,
        text_type="query",
    )[0]

    query_result: dict[str, Any] = collection.query(
        query_embeddings=[question_embedding],
        n_results=result_limit,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    ids = query_result.get("ids", [[]])[0]
    documents = query_result.get("documents", [[]])[0]
    metadatas = query_result.get("metadatas", [[]])[0]
    distances = query_result.get("distances", [[]])[0]

    search_results: list[SearchResult] = []

    for (
        chunk_id,
        content,
        metadata,
        distance,
    ) in zip(
        ids,
        documents,
        metadatas,
        distances,
        strict=True,
    ):
        search_results.append(
            SearchResult(
                chunk_id=chunk_id,
                document_name=str(
                    metadata["document_name"]
                ),
                page_number=int(
                    metadata["page_number"]
                ),
                content=str(content),
                distance=float(distance),
            )
        )

    return search_results


def print_search_results(
    question: str,
    results: list[SearchResult],
) -> None:
    """
    Exibe os resultados encontrados no terminal.
    """
    print()
    print("=" * 70)
    print(f"Pergunta: {question}")
    print("=" * 70)

    for position, result in enumerate(
        results,
        start=1,
    ):
        print()
        print(f"Resultado {position}")
        print(f"Documento: {result.document_name}")
        print(f"Página: {result.page_number}")
        print(f"ID: {result.chunk_id}")
        print(f"Distância: {result.distance:.4f}")
        print("-" * 70)
        print(result.content[:700])


def main() -> None:
    """
    Reconstrói a base e executa uma busca semântica de teste.
    """
    stored_count = build_vector_store()

    if stored_count <= 0:
        raise RuntimeError(
            "A base vetorial foi criada sem conteúdo."
        )

    test_question = (
        "Qual é o prazo para trocar uma roupa "
        "por outro tamanho?"
    )

    model = create_embedding_model()
    client = create_chroma_client()
    collection = get_or_create_collection(client)

    results = search_similar_chunks(
        question=test_question,
        result_limit=5,
        model=model,
        collection=collection,
    )

    print_search_results(
        question=test_question,
        results=results,
    )


if __name__ == "__main__":
    main()