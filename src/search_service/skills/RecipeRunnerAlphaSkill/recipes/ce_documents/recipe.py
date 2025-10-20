from ...models.recipes import ProcessingRecipe, MultiProcessingRecipe
from ...models.constants import (
    FileType,
    FileProcessorName,
    ChunkingUnit,
    ErrorHandlingStrategy,
)
from ...models.processing_controls import ADLSProcessingControl
from ...specs.datasource_specs.oai_datasource_spec import adls_data_source_spec
from ...models.chunking import ChunkingStrategy

ms_word_processing_recipe = ProcessingRecipe(
    file_types=[FileType.DOC, FileType.DOCX],
    file_processor_name=FileProcessorName.MS_WORD_PAGES_PROCESSOR,
    chunking_strategy=ChunkingStrategy(
        size=1200, overlap=200, unit=ChunkingUnit.TOKENS
    ),
    error_handling=ErrorHandlingStrategy.SKIP,
)

proccessing_control = ADLSProcessingControl(
    adls_data_source_spec=adls_data_source_spec,
    included_directories=["ce-documents"],
    excluded_directories=[]
)

ce_documents_processing_recipe = MultiProcessingRecipe(
    id="recipe_010",
    recipes=[ms_word_processing_recipe],
    processing_control=proccessing_control,
)
