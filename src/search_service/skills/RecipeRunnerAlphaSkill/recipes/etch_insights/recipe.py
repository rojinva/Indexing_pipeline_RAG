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


pptx_processing_recipe = ProcessingRecipe(
    file_types=[FileType.PPTX],
    file_processor_name=FileProcessorName.PRESENTATION_PPTX_PROCESSOR,
    chunking_strategy=ChunkingStrategy(
        size=1200, overlap=200, unit=ChunkingUnit.TOKENS
    ),
    error_handling=ErrorHandlingStrategy.SKIP,
)

proccessing_control = ADLSProcessingControl(
    adls_data_source_spec=adls_data_source_spec,
    included_directories=["EtchInsights"],
    excluded_directories=[],
)

etch_insights_processing_recipe = MultiProcessingRecipe(
    id="recipe_003",
    recipes=[pptx_processing_recipe],
    processing_control=proccessing_control,
)
