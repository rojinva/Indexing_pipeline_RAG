from pydantic import BaseModel

class StorageMetadata(BaseModel):
    parent_filename: str
    blob_uri: str

class SkillRequest(BaseModel):
    recordId: str
    data: StorageMetadata
