from pydantic import BaseModel, Field
from .constants import ChunkingUnit

class ChunkingStrategy(BaseModel):
    size: int = Field(..., description="Size of the chunk")
    overlap: int = Field(..., description="Overlap between chunks")
    unit: ChunkingUnit = Field(..., description="Unit of the chunk size and overlap")