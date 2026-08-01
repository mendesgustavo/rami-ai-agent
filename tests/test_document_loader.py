from app.document_loader import (
    DEFAULT_PDF_DIRECTORY,
    extract_pages_from_pdf,
    find_pdf_files,
    load_all_documents,
)


def test_find_five_pdf_files() -> None:
    pdf_files = find_pdf_files()

    assert len(pdf_files) == 5
    assert all(pdf_file.suffix.lower() == ".pdf" for pdf_file in pdf_files)


def test_extract_pages_from_first_pdf() -> None:
    first_pdf = find_pdf_files()[0]
    pages = extract_pages_from_pdf(first_pdf)

    assert len(pages) > 0
    assert all(page.content.strip() for page in pages)
    assert all(page.page_number > 0 for page in pages)


def test_load_all_documents() -> None:
    pages = load_all_documents()

    document_names = {
        page.document_name for page in pages
    }

    assert len(document_names) == 5
    assert len(pages) > 0


def test_expected_documents_exist() -> None:
    expected_documents = {
        "faq.pdf",
        "guia_de_envios_e_entregas.pdf",
        "politica_de_privacidade.pdf",
        "politica_de_reembolso_e_devolucoes.pdf",
        "termos_e_condicoes.pdf",
    }

    existing_documents = {
        path.name
        for path in DEFAULT_PDF_DIRECTORY.glob("*.pdf")
    }

    assert existing_documents == expected_documents