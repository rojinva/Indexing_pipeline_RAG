from typing import List
from pydantic import BaseModel, Field
from .chunking import ChunkingStrategy
from .constants import ErrorHandlingStrategy, FileProcessorName, FileType
from .processing_controls import ADLSProcessingControl


class ProcessingRecipe(BaseModel):
    file_types: List[FileType] = Field(
        ..., description="List of file types to be processed"
    )
    file_processor_name: FileProcessorName = Field(
        ..., description="Name of Processor to be used for the file"
    )
    chunking_strategy: ChunkingStrategy = Field(
        ..., description="Chunking strategy to be used for the file"
    )
    error_handling: ErrorHandlingStrategy = Field(
        ..., description="Error handling strategy to be used for the file"
    )


class MultiProcessingRecipe(BaseModel):
    id: str = Field(..., description="Unique identifier for the recipe.")
    recipes: List[ProcessingRecipe] = Field(
        ..., description="A list of processing recipes defining file processing rules."
    )
    processing_control: ADLSProcessingControl = Field(
        ...,
        description="The ADLS-specific processing control settings, including the data source.",
    )
