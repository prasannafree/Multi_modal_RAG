"""
Multimodal RAG Parser Module for LLM Applications.

Provides structured, simple, and robust parsers for:
- PDF (.pdf)
- Word Documents (.docx)
- Text Files (.txt)
- Markdown Files (.md)
- Images (.png, .jpg, .jpeg, .webp, etc.)

Fulfills the standard RAG parsing workflow:
Document -> Extract Text / Tables (Markdown) / Images -> Preserve Layout Order -> Return Structured Documents
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import fitz  # PyMuPDF
from PIL import Image as PILImage


@dataclass
class Document:
    """
    Standard RAG Document representation compatible with LLM/RAG pipelines 
    (LangChain, LlamaIndex, Vector Databases).
    
    Attributes:
        page_content: Text block, Markdown table, or image reference string/description.
        metadata: Metadata containing page number, element_type, bbox, source path, layout order, etc.
    """
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_content": self.page_content,
            "metadata": self.metadata,
        }



# ----------main parsing functions--------------------#

def parse_pdf(
    file_path: Union[str, Path],
    output_image_dir: Optional[Union[str, Path]] = None,
    extract_tables: bool = True,
    extract_images: bool = True,
    to_dict: bool = False,) -> Union[List[Document], List[Dict[str, Any]]]:
    """
    union -> allow anyone one of the  python type mentioned inside it 
    optional -> means the variable can be a specific type or can be None 
    """
    path = Path(file_path)  # convert the file path to a Path object 
    if not path.exists():   # if the file path does not exist
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if output_image_dir:                                           # if user passes a dir_path
        Path(output_image_dir).mkdir(parents=True, exist_ok=True)  # create the output image directory

    try:
        doc = fitz.open(str(path))   # open the pdf file using fitz 
    except Exception as e:
        raise ValueError(f"Failed to open PDF document '{file_path}': {e}") from e

    documents: List[Document] = []         # declaring the variable called documents which is a list of Document objects
    abs_path_str = str(path.resolve())     # get the absolute path of the file in string datatype .

    for page_idx, page in enumerate(doc):   # iterate through every page of the document
        page_num = page_idx + 1             # adding  page number to be included in metadata
        page_elements = []        # place holder for page elements
        table_bboxes = []         # placeholder for table bounding boxes

        # 1. Extract Tables (as Markdown)
        if extract_tables:      # extracts the table elements in the page as adds to the page_element holder
            try:
                for tab_idx, tab in enumerate(page.find_tables()):
                    table_bboxes.append(tab.bbox)
                    rows = tab.extract()
                    md_table = _rows_to_markdown(rows)
                    x0, y0, x1, y1 = tab.bbox

                    doc_item = Document(
                        page_content=md_table,
                        metadata={
                            "source": abs_path_str,
                            "page": page_num,
                            "element_type": "table",
                            "bbox": [x0, y0, x1, y1],
                            "table_index": tab_idx + 1,
                            "rows_count": len(rows),
                            "cols_count": len(rows[0]) if rows else 0,
                        },
                    )
                    page_elements.append((y0, x0, doc_item))
            except Exception as e:
                print(f"[Warning] Table extraction failed on page {page_num}: {e}")

        # 2. Extract Text Blocks
        for block in page.get_text("blocks"):   # extracts the text blocks along with position and adds that to page_element holder
            if len(block) >= 7 and block[6] == 0:  # Text block type in PyMuPDF
                x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
                cleaned_text = text.strip()
                if not cleaned_text:
                    continue

                if _is_inside_tables((x0, y0, x1, y1), table_bboxes):
                    continue

                doc_item = Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": abs_path_str,
                        "page": page_num,
                        "element_type": "text",
                        "bbox": [x0, y0, x1, y1],
                        "block_no": block[5],
                    },
                )
                page_elements.append((y0, x0, doc_item))

        # 3. Extract Embedded Images
        if extract_images:      # extracts embedded images and adds them to page_elements holder 
            try:
                for img_idx, img_info in enumerate(page.get_images(full=True)):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    saved_path = None

                    if output_image_dir:
                        img_name = f"page_{page_num}_img_{img_idx + 1}_{xref}.{base_image['ext']}"
                        save_file = Path(output_image_dir) / img_name
                        save_file.write_bytes(base_image["image"])
                        saved_path = str(save_file)

                    doc_item = Document(
                        page_content=f"[Image: Page {page_num}, Index {img_idx + 1}]",
                        metadata={
                            "source": abs_path_str,
                            "page": page_num,
                            "element_type": "image",
                            "bbox": None,
                            "xref": xref,
                            "ext": base_image["ext"],
                            "width": base_image.get("width"),
                            "height": base_image.get("height"),
                            "image_path": saved_path,
                        },
                    )
                    page_elements.append((9999 + img_idx, 0, doc_item))
            except Exception as e:
                print(f"[Warning] Image extraction failed on page {page_num}: {e}")

        # 4. PRESERVE LAYOUT: Sort elements by top-to-bottom spatial coordinate (y0, x0)
        page_elements.sort(key=lambda item: (item[0], item[1]))

        # 5. Assign layout reading order
        for layout_idx, (_, _, doc_item) in enumerate(page_elements, start=1):
            doc_item.metadata["layout_order"] = layout_idx
            documents.append(doc_item)

    doc.close()

    if to_dict:
        return [d.to_dict() for d in documents]
    return documents


def parse_docx(
    file_path: Union[str, Path],
    output_image_dir: Optional[Union[str, Path]] = None,
    extract_tables: bool = True,
    extract_images: bool = True,
    to_dict: bool = False,) -> Union[List[Document], List[Dict[str, Any]]]:
    """
    Parses a DOCX Microsoft Word file into structured RAG Document objects,
    extracting text paragraphs, Markdown tables, and embedded images.
    """
    try:
        import docx
    except ImportError:
        raise ImportError("python-docx package is required for docx parsing. Install it via 'pip install python-docx'.")

    path = Path(file_path)   # getting the file path of thay document 
    if not path.exists():
        raise FileNotFoundError(f"DOCX file not found: {file_path}")

    abs_path_str = str(path.resolve())    # getting the abs path of that document in string data type 
    doc = docx.Document(str(path))        # opening the word document
    documents: List[Document] = []        # initializing the documents list 
    layout_idx = 1                      # initializing the layout index

    if output_image_dir:
        Path(output_image_dir).mkdir(parents=True, exist_ok=True)   # if user give  the dir , then create a dir in that workspace . 

    # Parse elements in document reading order
    for block in doc.element.body:
        if block.tag.endswith("p"):
            p = docx.text.paragraph.Paragraph(block, doc)
            cleaned_text = p.text.strip()
            if cleaned_text:
                style_name = p.style.name if p.style else "Normal"
                is_heading = style_name.startswith("Heading")
                doc_item = Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": abs_path_str,
                        "element_type": "heading" if is_heading else "text",
                        "style": style_name,
                        "layout_order": layout_idx,
                    },
                )
                documents.append(doc_item)
                layout_idx += 1

        elif block.tag.endswith("tbl") and extract_tables:   # extracts table elements and add it in page_element holder 
            table = docx.table.Table(block, doc)
            rows_data = []
            for row in table.rows:
                rows_data.append([cell.text for cell in row.cells])

            md_table = _rows_to_markdown(rows_data)
            if md_table:
                doc_item = Document(
                    page_content=md_table,
                    metadata={
                        "source": abs_path_str,
                        "element_type": "table",
                        "rows_count": len(table.rows),
                        "cols_count": len(table.columns) if table.rows else 0,
                        "layout_order": layout_idx,
                    },
                )
                documents.append(doc_item)
                layout_idx += 1

    # Extract embedded images from document relationship parts
    if extract_images and output_image_dir:
        try:
            for rel_id, rel in doc.part.rels.items():
                if "image" in rel.target_ref:
                    img_part = rel.target_part
                    img_bytes = img_part.blob
                    ext = img_part.content_type.split("/")[-1]
                    img_filename = f"docx_img_{rel_id}.{ext}"
                    save_file = Path(output_image_dir) / img_filename
                    save_file.write_bytes(img_bytes)

                    doc_item = Document(
                        page_content=f"[Image: DOCX Embedded Image {rel_id}]",
                        metadata={
                            "source": abs_path_str,
                            "element_type": "image",
                            "image_path": str(save_file),
                            "layout_order": layout_idx,
                        },
                    )
                    documents.append(doc_item)
                    layout_idx += 1
        except Exception as e:
            print(f"[Warning] DOCX image extraction warning: {e}")

    if to_dict:
        return [d.to_dict() for d in documents]
    return documents


def parse_txt(
    file_path: Union[str, Path],
    chunk_by_paragraph: bool = True,
    to_dict: bool = False,) -> Union[List[Document], List[Dict[str, Any]]]:
    """
    Parses a plain text (.txt) file into structured RAG Document objects.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"TXT file not found: {file_path}")

    abs_path_str = str(path.resolve())
    raw_text = path.read_text(encoding="utf-8", errors="replace")

    documents: List[Document] = []

    if chunk_by_paragraph:
        paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
        for idx, p in enumerate(paragraphs, start=1):
            doc_item = Document(
                page_content=p,
                metadata={
                    "source": abs_path_str,
                    "element_type": "text",
                    "layout_order": idx,
                },
            )
            documents.append(doc_item)
    else:
        documents.append(
            Document(
                page_content=raw_text.strip(),
                metadata={
                    "source": abs_path_str,
                    "element_type": "text",
                    "layout_order": 1,
                    "file_size_bytes": path.stat().st_size,
                },
            )
        )

    if to_dict:
        return [d.to_dict() for d in documents]
    return documents


def parse_md(
    file_path: Union[str, Path],
    chunk_by_section: bool = True,
    to_dict: bool = False,) -> Union[List[Document], List[Dict[str, Any]]]:
    """
    Parses a Markdown (.md) file into structured RAG Document objects.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Markdown file not found: {file_path}")

    abs_path_str = str(path.resolve())
    raw_text = path.read_text(encoding="utf-8", errors="replace")

    documents: List[Document] = []

    if chunk_by_section:
        blocks = [b.strip() for b in raw_text.split("\n\n") if b.strip()]
        for idx, block in enumerate(blocks, start=1):
            is_header = block.startswith("#")
            doc_item = Document(
                page_content=block,
                metadata={
                    "source": abs_path_str,
                    "element_type": "header" if is_header else "text",
                    "layout_order": idx,
                },
            )
            documents.append(doc_item)
    else:
        documents.append(
            Document(
                page_content=raw_text.strip(),
                metadata={
                    "source": abs_path_str,
                    "element_type": "text",
                    "layout_order": 1,
                    "file_size_bytes": path.stat().st_size,
                },
            )
        )

    if to_dict:
        return [d.to_dict() for d in documents]
    return documents


def parse_image(
    file_path: Union[str, Path],
    description: Optional[str] = None,
    to_dict: bool = False,) -> Union[Document, Dict[str, Any]]:
    """
    Parses a standalone image file (PNG, JPG, WEBP, BMP, TIFF, etc.) into a structured RAG Document.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    abs_path_str = str(path.resolve())

    try:
        with PILImage.open(path) as img:
            width, height = img.size
            img_format = img.format or path.suffix.lstrip(".").lower()
            img_mode = img.mode
    except Exception as e:
        raise ValueError(f"Failed to open image file '{file_path}': {e}") from e

    content = description or f"[Image File: {path.name} | Resolution: {width}x{height} | Format: {img_format.upper()}]"

    doc_item = Document(
        page_content=content,
        metadata={
            "source": abs_path_str,
            "element_type": "image",
            "format": str(img_format).lower(),
            "width": width,
            "height": height,
            "mode": img_mode,
            "file_size_bytes": path.stat().st_size,
        },
    )

    if to_dict:
        return doc_item.to_dict()
    return doc_item


def parse_file(
    file_path: Union[str, Path],
    output_image_dir: Optional[Union[str, Path]] = None,
    to_dict: bool = False,) -> Union[List[Document], List[Dict[str, Any]], Document, Dict[str, Any]]:
    """
    Unified file parser dispatcher routing PDF, DOCX, TXT, MD, or Image files 
    to their respective parsers.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return parse_pdf(file_path, output_image_dir=output_image_dir, to_dict=to_dict)
    elif suffix in [".docx", ".doc"]:
        return parse_docx(file_path, output_image_dir=output_image_dir, to_dict=to_dict)
    elif suffix == ".txt":
        return parse_txt(file_path, to_dict=to_dict)
    elif suffix in [".md", ".markdown"]:
        return parse_md(file_path, to_dict=to_dict)
    elif suffix in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
        return parse_image(file_path, to_dict=to_dict)
    else:
        raise ValueError(f"Unsupported file format '{suffix}' for path: {file_path}")



#------------------supporting function -------------------#

def _rows_to_markdown(rows: List[List[Optional[str]]]) -> str:
    """Converts 2D table array into Markdown table string."""
    if not rows:
        return ""
    cleaned = [[(c.replace("\n", " ").strip() if c else "") for c in r] for r in rows]
    header = cleaned[0]
    md = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in cleaned[1:]:
        r = r[: len(header)] + [""] * max(0, len(header) - len(r))
        md.append("| " + " | ".join(r) + " |")
    return "\n".join(md)


def _is_inside_tables(bbox: tuple, table_bboxes: list) -> bool:
    """Checks if bounding box is within an extracted table region."""
    x0, y0, x1, y1 = bbox
    for tx0, ty0, tx1, ty1 in table_bboxes:
        if x0 >= tx0 - 2 and y0 >= ty0 - 2 and x1 <= tx1 + 2 and y1 <= ty1 + 2:
            return True
    return False
