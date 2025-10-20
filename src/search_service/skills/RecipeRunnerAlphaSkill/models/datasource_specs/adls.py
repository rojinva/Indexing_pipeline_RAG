import os
from typing import Optional
from pydantic import BaseModel, root_validator, Field

from dotenv import load_dotenv
load_dotenv(override=True)

class AuthSecretSpec(BaseModel):
    secret_suffix: str = Field(
        ...,
        description="The suffix of the secret to use when reading credentials from environment variables. "
        "For example, if the secret_name is 'MYAPP_SECRET', then the expected environment "
        "variables are 'TENANT_ID_MYAPP_SECRET', 'CLIENT_ID_MYAPP_SECRET', and 'CLIENT_SECRET_MYAPP_SECRET'.",
    )
    tenant_id: Optional[str] = Field(
        None,
        description="Tenant ID for Azure authentication. Automatically loaded from environment variables based on secret_suffix.",
    )
    client_id: Optional[str] = Field(
        None,
        description="Client ID for Azure authentication. Automatically loaded from environment variables based on secret_suffix.",
    )
    client_secret: Optional[str] = Field(
        None,
        description="Client secret for Azure authentication. Automatically loaded from environment variables based on secret_suffix.",
    )

    @root_validator(pre=True)
    def load_credentials_from_env(cls, values):
        """
        Dynamically load credentials from environment variables based on the secret_suffix.
        Expected environment variable names are:
            TENANT_ID_{SECRET_SUFFIX}
            CLIENT_ID_{SECRET_SUFFIX}
            CLIENT_SECRET_{SECRET_SUFFIX}
        """
        secret_suffix = values.get("secret_suffix")
        print(f" secret_suffix: {secret_suffix}")
        if secret_suffix:
            values["tenant_id"] = os.environ.get(f"TENANT_ID_{secret_suffix}")
            values["client_id"] = os.environ.get(f"CLIENT_ID_{secret_suffix}")
            values["client_secret"] = os.environ.get(f"CLIENT_SECRET_{secret_suffix}")
        return values

    def get_credentials(self):
        return self.tenant_id, self.client_id, self.client_secret


class ADLSDataSourceSpec(BaseModel):
    storage_account_name: str = Field(
        ..., description="The name of the Azure Blob Storage account to connect to."
    )
    container_name: str = Field(
        ...,
        description="The name of the container within the Azure Blob Storage account.",
    )
    auth_secret: AuthSecretSpec = Field(
        ...,
        description="The authentication specification containing the credentials for Azure.",
    )
