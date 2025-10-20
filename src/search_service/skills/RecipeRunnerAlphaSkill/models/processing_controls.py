from typing import List, Optional
from pydantic import BaseModel, Field
from .datasource_specs.adls import ADLSDataSourceSpec


class ADLSProcessingControl(BaseModel):
    adls_data_source_spec: ADLSDataSourceSpec = Field(
        ...,
        description="The ADLS data source specification associated with this processing control.",
    )
    included_directories: Optional[List[str]] = Field(
        None, description="List of directories to include in processing."
    )
    excluded_directories: Optional[List[str]] = Field(
        None, description="List of directories to exclude from processing."
    )