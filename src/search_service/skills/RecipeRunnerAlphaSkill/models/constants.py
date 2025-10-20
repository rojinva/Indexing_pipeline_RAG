import enum

class StrEnum(str, enum.Enum):
    pass

class FileType(StrEnum):
    PDF = "pdf"
    DOC = "doc"
    DOCX = "docx"
    PPT = "ppt"
    PPTX = "pptx"
    TXT = "txt"
    HTML = "html"

class ChunkingUnit(StrEnum):
    TOKENS = "tokens"

class ErrorHandlingStrategy(StrEnum):
    RETRY = "retry"
    SKIP = "skip"
    LOG = "log"

class FileProcessorName(StrEnum):
    FINANCE_PDF_PROCESSOR = "finance_pdf_processor"
    PDF_PAGE_MARKDOWN_PROCESSOR = "pdf_page_markdown_processor"
    MS_WORD_PAGES_PROCESSOR = "ms_word_pages_processor"
    MS_WORD_LLM_PAGES_PROCESSOR = "ms_word_llm_pages_processor"
    TRANSCRIPTS_MS_WORD_PAGES_PROCESSOR = "transcripts_ms_word_pages_processor"
    FINANCE_PDF_PAGE_MARKDOWN_PROCESSOR = "finance_pdf_page_markdown_processor"
    PRESENTATION_PPTX_PROCESSOR = "presentation_pptx_processor"

class ResponseModelType(StrEnum):
    MULTIMODAL = "multimodal"
    FINANCE = "finance"

class RemainderMode(StrEnum):
    passthrough = "passthrough"
    ignore = "ignore"

class SourceModality(StrEnum):
    IMAGE = "image"
    TEXT = "text"