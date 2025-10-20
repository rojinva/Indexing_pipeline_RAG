import asyncio
from typing import List, Any, Dict
from pydantic import BaseModel
from .models.request import SkillRequest
from .models.constants import FileProcessorName
from .models.recipes import MultiProcessingRecipe
from .models.constants import FileProcessorName, ChunkingUnit
from .utils.file_type_identifier import FileTypeIdentifier
from .clients.azure_blob_service_client import AzureBlobServiceClient
from .utils.string_parsing import parse_blob_uri_root_directory_name
from .registry import RECIPE_REGISTRY, PROCESSOR_REGISTRY
from langchain_text_splitters import TokenTextSplitter


class RecipeRunner(BaseModel):
    recipe_id: str
    skill_requests: List[SkillRequest]

    def __init__(self, recipe_id: str, skill_requests: List[SkillRequest]):
        super().__init__(recipe_id=recipe_id, skill_requests=skill_requests)

    @staticmethod
    def _get_recipe(id: str):
        return RECIPE_REGISTRY.get(id)

    @staticmethod
    def _get_processor(name: FileProcessorName):
        return PROCESSOR_REGISTRY.get(name)

    async def _process_single(self, request: SkillRequest) -> Dict[str, Any]:
        """
        Process a single file using the FinancePDFProcessor so that the output matches the structure:
          {
            "recordId": record_id,
            "data": {"chunks": chunks_with_metadata},
            "errors": error_list,
            "warnings": warning
          }
        """
        # Generate a record ID (could be extracted from the request in a real scenario)
        record_id = request.recordId
        blob_uri = request.data.blob_uri
        error_list = []
        warning = []
        try:
            # Step 1: Identify the file type based on the blob URI.
            file_type_identifier = FileTypeIdentifier()
            identified_file_type = file_type_identifier.identify_file_type(
                blob_uri=blob_uri
            )

            processing_recipe: MultiProcessingRecipe = self._get_recipe(
                id=self.recipe_id
            )
            included_dirs = processing_recipe.processing_control.included_directories
            excluded_dirs = processing_recipe.processing_control.excluded_directories

            top_directory = parse_blob_uri_root_directory_name(blob_uri)
            print(f"[{record_id}] Top directory extracted: {top_directory}")

            # Check if the file comes from an excluded directory.
            if top_directory in excluded_dirs:
                warning.append(
                    f"File from excluded directory: {top_directory}. Skipping processing."
                )
                return {
                    "recordId": record_id,
                    "data": {"chunks": []},
                    "errors": error_list,
                    "warnings": warning,
                }

            # Check if the file comes from one of the included directories.
            if top_directory not in included_dirs:
                warning.append(
                    f"File not in included directories. Found directory: {top_directory}. Skipping processing."
                )
                return {
                    "recordId": record_id,
                    "data": {"chunks": []},
                    "errors": error_list,
                    "warnings": warning,
                }

            matching_recipe = None
            for recipe in processing_recipe.recipes:
                if (
                    identified_file_type in recipe.file_types
                ):  # file_types is a list of FileType values
                    matching_recipe = recipe

            if matching_recipe is None:
                raise ValueError(
                    f"No matching processing recipe found for file type: {identified_file_type.value}"
                )

            # Step 3: Get the file processor based on the recipe's processor name.
            processor_name = matching_recipe.file_processor_name
            processor_cls = self._get_processor(name=processor_name)
            if processor_cls is None:
                raise ValueError(f"Processor for {processor_name} is not defined.")

            # Create the data source client from the specification provided.
            datasource_spec = processing_recipe.processing_control.adls_data_source_spec
            datasource_client = AzureBlobServiceClient.from_spec(datasource_spec)

            if matching_recipe.chunking_strategy.unit == ChunkingUnit.TOKENS:
                # Use the text splitter for token-based chunking.
                text_splitter = TokenTextSplitter(
                    chunk_size=matching_recipe.chunking_strategy.size,
                    chunk_overlap=matching_recipe.chunking_strategy.overlap
                )

            # Step 5: Process the file.
            processor_instance = processor_cls()
            processed_chunks = await processor_instance.aprocess(
                storage_metadata=request.data,
                text_splitter=text_splitter,
                datasource_client=datasource_client,
            )

            data_output = {"chunks": [chunk.dict() for chunk in processed_chunks]}

            print(
                f"[{record_id}] Completed processing. Chunks: {len(processed_chunks)}"
            )
            return {
                "recordId": record_id,
                "data": data_output,
                "errors": error_list,
                "warnings": warning,
            }
        except Exception as e:
            error_list.append(f"Failed to process file: {str(e)}")
            return {
                "recordId": record_id,
                "data": {"chunks": []},
                "errors": error_list,
                "warnings": warning,
            }

    async def process_all(self) -> Dict[str, Any]:
        """
        Process all files in parallel and return the output in a JSON-ready dict with a "values" key containing a list of individual file processing results.
        """
        tasks = [self._process_single(request) for request in self.skill_requests]
        results = await asyncio.gather(*tasks)
        return {"values": results}
