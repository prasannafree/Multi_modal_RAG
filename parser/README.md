# Multimodal RAG Parser Module Documentation

The `parser` module provides structured, simple, and robust parsers for extracting text, tables (as Markdown), and images from diverse file formats (**PDF**, **DOCX**, **TXT**, **Markdown**, and **Images**) into standard RAG `Document` objects.

---

## 1. Class Defined

### `Document`
A dataclass representing the standardized output format for all extracted document elements, designed to integrate directly with RAG frameworks (LangChain, LlamaIndex, Vector Databases).

* **Attributes**:
  * `page_content` (`str`): The extracted text block, Markdown table string, or image description/reference.
  * `metadata` (`Dict[str, Any]`): Context dictionary storing details such as `source` file path, `page` number, `element_type` (`"text"`, `"table"`, `"image"`, `"heading"`), `bbox`, and `layout_order`.
* **Methods**:
  * `to_dict()`: Converts the `Document` object into a plain Python dictionary (`{"page_content": ..., "metadata": ...}`).

---

## 2. Main Parsing Functions

### 1. `parse_pdf()`
Parses PDF documents into structured `Document` objects while preserving reading layout order.

* **Inputs**:
  * `file_path` (`Union[str, Path]`): Target PDF file path.
  * `output_image_dir` (`Optional[Union[str, Path]]`, default `None`): Directory path to save extracted images.
  * `extract_tables` (`bool`, default `True`): Whether to detect and extract tables as Markdown.
  * `extract_images` (`bool`, default `True`): Whether to extract embedded image assets.
  * `to_dict` (`bool`, default `False`): If `True`, returns a list of dictionaries instead of `Document` objects.
* **Outputs**:
  * `Union[List[Document], List[Dict[str, Any]]]`: List of structured `Document` items ordered by top-to-bottom page layout.
* **Steps Followed**:
  1. Validate file existence and create `output_image_dir` if specified.
  2. Open PDF document using PyMuPDF (`fitz.open()`).
  3. For each page in the document:
     a. **Extract Tables**: Use `page.find_tables()`, extract 2D row arrays, convert to Markdown via `_rows_to_markdown()`, and record table bounding boxes (`bbox`).
     b. **Extract Text Blocks**: Extract blocks via `page.get_text("blocks")`. Filter out empty blocks and text falling inside detected table bounding boxes (`_is_inside_tables()`).
     c. **Extract Images**: Iterate through page images (`page.get_images()`), extract raw image bytes, and save image files if `output_image_dir` is provided.
     d. **Preserve Layout Order**: Sort all extracted page elements (tables, text, images) by spatial vertical coordinate `(y0, x0)`.
     e. **Assign Metadata**: Attach sequential `layout_order` indices per page.
  4. Return the complete list of `Document` objects.

---

### 2. `parse_docx()`
Parses Microsoft Word (`.docx`) files into structured `Document` objects.

* **Inputs**:
  * `file_path` (`Union[str, Path]`): Target `.docx` file path.
  * `output_image_dir` (`Optional[Union[str, Path]]`, default `None`): Folder path to save extracted images.
  * `extract_tables` (`bool`, default `True`): Whether to convert Word tables to Markdown.
  * `extract_images` (`bool`, default `True`): Whether to extract embedded images.
  * `to_dict` (`bool`, default `False`): If `True`, returns dictionaries instead of `Document` objects.
* **Outputs**:
  * `Union[List[Document], List[Dict[str, Any]]]`: List of structured elements preserving Word document flow.
* **Steps Followed**:
  1. Verify `python-docx` installation and file existence.
  2. Open document via `docx.Document(file_path)`.
  3. Iterate sequentially through document body XML blocks (`doc.element.body`):
     a. **Paragraphs (`p`)**: Extract text, detect heading style (`Heading 1`, `Heading 2`), and append as text/heading `Document`.
     b. **Tables (`tbl`)**: Extract row cell strings, convert to Markdown, and append as table `Document`.
  4. **Embedded Images**: Iterate through document relationships (`doc.part.rels`), extract image binary blobs, and save files to `output_image_dir`.
  5. Assign sequential `layout_order` metadata and return.

---

### 3. `parse_txt()`
Parses plain text (`.txt`) files into structured `Document` objects.

* **Inputs**:
  * `file_path` (`Union[str, Path]`): Target `.txt` file path.
  * `chunk_by_paragraph` (`bool`, default `True`): If `True`, splits content by double line breaks (`\n\n`).
  * `to_dict` (`bool`, default `False`): If `True`, returns dicts instead of `Document` objects.
* **Outputs**:
  * `Union[List[Document], List[Dict[str, Any]]]`: List of text `Document` items.
* **Steps Followed**:
  1. Check file existence.
  2. Read raw UTF-8 content from text file.
  3. If `chunk_by_paragraph` is enabled, split text by `\n\n` into paragraph chunks.
  4. Create `Document` objects with layout ordering and return.

---

### 4. `parse_md()`
Parses Markdown (`.md`, `.markdown`) files into structured `Document` objects.

* **Inputs**:
  * `file_path` (`Union[str, Path]`): Target `.md` file path.
  * `chunk_by_section` (`bool`, default `True`): If `True`, splits content by section blocks.
  * `to_dict` (`bool`, default `False`): If `True`, returns dicts instead of `Document` objects.
* **Outputs**:
  * `Union[List[Document], List[Dict[str, Any]]]`: List of Markdown `Document` items.
* **Steps Followed**:
  1. Check file existence.
  2. Read raw UTF-8 content.
  3. Split text by double line breaks (`\n\n`).
  4. Differentiate header blocks (`starts with #`) from body text (`element_type: "header"` vs `"text"`).
  5. Assign `layout_order` metadata and return.

---

### 5. `parse_image()`
Parses standalone image files (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`) into a structured `Document`.

* **Inputs**:
  * `file_path` (`Union[str, Path]`): Target image path.
  * `description` (`Optional[str]`, default `None`): Optional Vision LLM summary or caption text.
  * `to_dict` (`bool`, default `False`): If `True`, returns dict instead of `Document`.
* **Outputs**:
  * `Union[Document, Dict[str, Any]]`: Single `Document` representing the image asset.
* **Steps Followed**:
  1. Verify file existence.
  2. Open image using Pillow (`PIL.Image.open()`).
  3. Extract image metadata: width, height, image format, color mode, and file size.
  4. Use `description` string or construct default image reference tag `[Image File: ... | Resolution: WxH]`.
  5. Return `Document` object with metadata.

---

## 3. Supporting & Dispatcher Functions

### 1. `parse_file()`
Unified entry-point dispatcher for parsing any supported file type.

* **Inputs**: `file_path`, `output_image_dir` (optional), `to_dict` (optional).
* **Outputs**: Structured `Document` or `List[Document]`.
* **Steps Followed**:
  1. Inspect file extension (`.pdf`, `.docx`, `.txt`, `.md`, `.png`, etc.).
  2. Automatically route the file to `parse_pdf()`, `parse_docx()`, `parse_txt()`, `parse_md()`, or `parse_image()`.

---

### 2. `_rows_to_markdown()`
Helper function converting a 2D matrix of cell strings into a clean Markdown table format.

* **Inputs**: `rows` (`List[List[Optional[str]]]`).
* **Outputs**: `str` (Markdown table text).
* **Steps Followed**:
  1. Sanitize cell strings (strip newlines and extra spaces).
  2. Format row 0 as table header: `| Col 1 | Col 2 |`.
  3. Add Markdown separator row: `| --- | --- |`.
  4. Append remaining data rows with matching column padding.

---

### 3. `_is_inside_tables()`
Helper function checking spatial overlap between text block coordinates and table regions.

* **Inputs**: `bbox` (`tuple`), `table_bboxes` (`list`).
* **Outputs**: `bool` (`True` if block is inside a table).
* **Steps Followed**:
  1. Compare bounding box coordinates `(x0, y0, x1, y1)` against extracted table bounding boxes with a 2-pixel tolerance margin.
  2. Return `True` if contained within any table box to prevent text duplication.
