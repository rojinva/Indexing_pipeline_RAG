from pydantic import BaseModel, Field
from .constants import SourceModality

class BaseSearchIndexRecord(BaseModel):
    parent_path: str = Field(..., description="File path of the parent document")
    use_case: str = Field(..., description="Description of the use case for the document")
    content: str = Field(..., description="Text content of the chunk")
    chunk_num: int = Field(..., description="Number of the chunk")
    chunk_hash: str = Field(..., description="Hash value of the chunk for integrity verification")
    row: int = Field(..., description="Row number in the dataset")
    sheet_name: str = Field(..., description="Name of the sheet in the dataset")

class MultimodalSearchIndexRecord(BaseSearchIndexRecord):
    source_modality: SourceModality = Field(..., description="Modality of the source (e.g., text, image, video)")
    media_path: str = Field(..., description="Path to the media file in azure blob storage (e.g., image, video) associated with the chunk")

class FinanceSearchIndexRecord(BaseSearchIndexRecord):
    title: str = Field(..., description="The official title of the document, typically not the first line but a key identifier of its content.")
    contributing_authors: str = Field(
        ..., description="The names of all authors who contributed to the creation of the document."
    )
    publication_date: str = Field(..., description="Publication date of the document")
    company_name: str = Field(..., description="Name of the publication company.")