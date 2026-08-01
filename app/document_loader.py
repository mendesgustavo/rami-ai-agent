from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF_DIRECTORY = PROJECT_ROOT / "documents" / "pdf"


@dataclass
class DocumentPage:
    """
    Representa uma página extraída de um documento PDF.
    """

    document_name: str
    page_number: int
    content: str


def find_pdf_files(directory: Path = DEFAULT_PDF_DIRECTORY) -> list[Path]:
    """
    Localiza todos os arquivos PDF existentes no diretório informado.
    """
    if not directory.exists():
        raise FileNotFoundError(
            f"O diretório de PDFs não foi encontrado: {directory}"
        )

    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"Nenhum arquivo PDF foi encontrado em: {directory}"
        )

    return pdf_files


def extract_pages_from_pdf(pdf_path: Path) -> list[DocumentPage]:
    """
    Extrai o texto de todas as páginas de um arquivo PDF.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"O arquivo PDF não foi encontrado: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"O arquivo informado não é um PDF: {pdf_path}"
        )

    reader = PdfReader(str(pdf_path))
    extracted_pages: list[DocumentPage] = []

    for page_index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_text = page_text.strip()

        if not page_text:
            continue

        extracted_pages.append(
            DocumentPage(
                document_name=pdf_path.name,
                page_number=page_index,
                content=page_text,
            )
        )

    return extracted_pages


def load_all_documents(
    directory: Path = DEFAULT_PDF_DIRECTORY,
) -> list[DocumentPage]:
    """
    Localiza todos os PDFs e extrai as páginas que possuem texto.
    """
    all_pages: list[DocumentPage] = []

    for pdf_file in find_pdf_files(directory):
        pages = extract_pages_from_pdf(pdf_file)
        all_pages.extend(pages)

    return all_pages


def print_loading_summary(pages: list[DocumentPage]) -> None:
    """
    Exibe um resumo dos documentos carregados.
    """
    document_names = sorted(
        {page.document_name for page in pages}
    )

    total_characters = sum(
        len(page.content) for page in pages
    )

    print("Resumo do carregamento")
    print("----------------------")
    print(f"Documentos carregados: {len(document_names)}")
    print(f"Páginas com texto: {len(pages)}")
    print(f"Caracteres extraídos: {total_characters}")
    print()

    for document_name in document_names:
        document_pages = [
            page
            for page in pages
            if page.document_name == document_name
        ]

        print(
            f"- {document_name}: "
            f"{len(document_pages)} páginas com texto"
        )


def main() -> None:
    """
    Executa uma validação manual da leitura dos documentos.
    """
    pages = load_all_documents()
    print_loading_summary(pages)

    print("\nAmostra do primeiro conteúdo extraído")
    print("-------------------------------------")

    first_page = pages[0]

    print(f"Documento: {first_page.document_name}")
    print(f"Página: {first_page.page_number}")
    print()
    print(first_page.content[:800])


if __name__ == "__main__":
    main()