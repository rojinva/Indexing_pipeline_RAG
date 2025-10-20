from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union
from .records import MultimodalSearchIndexRecord, FinanceSearchIndexRecord

class Warning(BaseModel):
    """Represents a warning message related to processing."""
    message: str = Field(..., description="Warning message detailing any issues encountered during processing.")

class ProcessedFile(BaseModel):
    """Represents the result of processing a single file, including its ID, errors, and warnings."""
    recordId: str = Field(..., description="Unique identifier for the file.")
    data: Dict[str, Union[FinanceSearchIndexRecord, MultimodalSearchIndexRecord]] = Field(..., description="Processed data for the file, including chunks.")
    errors: List[str] = Field(..., description="List of error messages encountered during processing.")
    warnings: List[Warning] = Field(..., description="List of warnings encountered during processing.")

class SkillResponse(BaseModel):
    """Represents the overall response expected by the search service, containing multiple file processing results."""
    values: List[ProcessedFile] = Field(..., description="List of file processing results for each processed file.")