import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "documents" / "source"
OUTPUT_DIR = PROJECT_ROOT / "documents" / "pdf"


def create_styles() -> dict:
    """Cria os estilos usados nos documentos PDF."""
    base_styles = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "RamiTitle",
            parent=base_styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=25,
            alignment=TA_CENTER,
            spaceAfter=20,
        ),
        "heading1": ParagraphStyle(
            "RamiHeading1",
            parent=base_styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            spaceBefore=14,
            spaceAfter=8,
        ),
        "heading2": ParagraphStyle(
            "RamiHeading2",
            parent=base_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            spaceBefore=10,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "RamiBody",
            parent=base_styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=15,
            spaceAfter=8,
        ),
        "metadata": ParagraphStyle(
            "RamiMetadata",
            parent=base_styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            leading=13,
            alignment=TA_CENTER,
            spaceAfter=5,
        ),
    }


def add_page_number(canvas, document) -> None:
    """Adiciona o número da página no rodapé."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)

    page_number = canvas.getPageNumber()
    page_width, _ = A4

    canvas.drawCentredString(
        page_width / 2,
        1.2 * cm,
        f"RAMI — Página {page_number}",
    )

    canvas.restoreState()


def format_inline_markdown(text: str) -> str:
    """
    Converte elementos simples de Markdown para tags compatíveis
    com o Paragraph do ReportLab.
    """
    # Converte links Markdown, como:
    # [atendimento@rami.com.br](mailto:atendimento@rami.com.br)
    # para:
    # atendimento@rami.com.br
    text = re.sub(r"\[([^\]]+)\]\((?:mailto:)?[^)]+\)", r"\1", text)

    safe_text = escape(text)

    parts = safe_text.split("**")

    for index in range(1, len(parts), 2):
        parts[index] = f"<b>{parts[index]}</b>"

    return "".join(parts)


def markdown_to_flowables(content: str, styles: dict) -> list:
    """Converte o conteúdo Markdown em elementos do ReportLab."""
    flowables = []
    bullet_items = []

    def flush_bullets() -> None:
        nonlocal bullet_items

        if not bullet_items:
            return

        list_items = [
            ListItem(
                Paragraph(item, styles["body"]),
                leftIndent=12,
            )
            for item in bullet_items
        ]

        flowables.append(
            ListFlowable(
                list_items,
                bulletType="bullet",
                leftIndent=20,
                bulletFontName="Helvetica",
                bulletFontSize=8,
            )
        )

        flowables.append(Spacer(1, 6))
        bullet_items = []

    for raw_line in content.splitlines():
        line = raw_line.strip()

        if not line:
            flush_bullets()
            flowables.append(Spacer(1, 5))
            continue

        if line.startswith("- "):
            bullet_items.append(format_inline_markdown(line[2:]))
            continue

        flush_bullets()

        if line.startswith("# "):
            flowables.append(
                Paragraph(
                    format_inline_markdown(line[2:]),
                    styles["title"],
                )
            )

        elif line.startswith("## "):
            flowables.append(
                Paragraph(
                    format_inline_markdown(line[3:]),
                    styles["heading1"],
                )
            )

        elif line.startswith("### "):
            flowables.append(
                Paragraph(
                    format_inline_markdown(line[4:]),
                    styles["heading2"],
                )
            )

        elif line.startswith("**Última atualização:**") or line.startswith(
            "**Versão:**"
        ):
            flowables.append(
                Paragraph(
                    format_inline_markdown(line),
                    styles["metadata"],
                )
            )

        elif line == "---":
            flowables.append(PageBreak())

        else:
            flowables.append(
                Paragraph(
                    format_inline_markdown(line),
                    styles["body"],
                )
            )

    flush_bullets()

    return flowables


def generate_pdf(source_file: Path, output_file: Path) -> None:
    """Gera um PDF a partir de um arquivo Markdown."""
    content = source_file.read_text(encoding="utf-8")
    styles = create_styles()
    flowables = markdown_to_flowables(content, styles)

    document = SimpleDocTemplate(
        str(output_file),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=source_file.stem.replace("_", " ").title(),
        author="RAMI",
        subject="Base de conhecimento do agente inteligente RAMI",
    )

    document.build(
        flowables,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )


def main() -> None:
    """Localiza os arquivos Markdown e gera todos os PDFs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    markdown_files = sorted(SOURCE_DIR.glob("*.md"))

    if not markdown_files:
        raise FileNotFoundError(
            f"Nenhum arquivo Markdown encontrado em: {SOURCE_DIR}"
        )

    print(f"Documentos encontrados: {len(markdown_files)}")

    for source_file in markdown_files:
        output_file = OUTPUT_DIR / f"{source_file.stem}.pdf"

        generate_pdf(source_file, output_file)

        print(f"PDF criado: {output_file.relative_to(PROJECT_ROOT)}")

    print("\nConversão concluída com sucesso.")


if __name__ == "__main__":
    main()