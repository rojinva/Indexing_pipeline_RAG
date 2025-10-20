from .recipes.cfpa.recipe import cfpa_processing_recipe
from .recipes.dpg_transcripts.recipe import dpg_transcripts_processing_recipe
from .recipes.etch_insights.recipe import etch_insights_processing_recipe
from .recipes.metrology.recipe import metrology_processing_recipe
from .recipes.cfpa_subset.recipe import cfpa_subset_processing_recipe
from .recipes.cfpa_subset.recipe_2 import cfpa_subset_processing_recipe_2
from .recipes.patent_insights.recipe import patent_processing_recipe
from .recipes.ethics_and_compliance.recipe import ethics_and_compliance_processing_recipe
from .processors.pdf_markdown_pages_processor import PDFPageMarkdownProcessor
from .processors.ms_word_pages_processor import MSWordProcessor
from .processors.ms_word_llm_pages_processor import MSWordLLMProcessor
from .processors.transcripts_ms_word_pages_processor import TranscriptsMSWordProcessor
from .processors.finance_pdf_markdown_pages_processor import FinancePDFMarkdownPagesProcessor
from .processors.presentation.presentation_pptx_processor import PresentationPPTXProcessor 
from .processors.finance.finance_pdf_processor import FinancePDFProcessor  
from .models.constants import FileProcessorName


RECIPE_REGISTRY = {
    "recipe_001": cfpa_processing_recipe,
    "recipe_002": dpg_transcripts_processing_recipe,
    "recipe_003": etch_insights_processing_recipe,
    "recipe_004": metrology_processing_recipe,
    "recipe_005": cfpa_subset_processing_recipe,
    "recipe_006": cfpa_subset_processing_recipe_2,
    "recipe_007": patent_processing_recipe,
    "recipe_011": ethics_and_compliance_processing_recipe
}

PROCESSOR_REGISTRY = {
    FileProcessorName.PDF_PAGE_MARKDOWN_PROCESSOR: PDFPageMarkdownProcessor,
    FileProcessorName.MS_WORD_PAGES_PROCESSOR: MSWordProcessor,
    FileProcessorName.MS_WORD_LLM_PAGES_PROCESSOR: MSWordLLMProcessor,
    FileProcessorName.TRANSCRIPTS_MS_WORD_PAGES_PROCESSOR: TranscriptsMSWordProcessor,
    FileProcessorName.FINANCE_PDF_PAGE_MARKDOWN_PROCESSOR: FinancePDFMarkdownPagesProcessor,
    FileProcessorName.PRESENTATION_PPTX_PROCESSOR: PresentationPPTXProcessor,
    FileProcessorName.FINANCE_PDF_PROCESSOR: FinancePDFProcessor
}
