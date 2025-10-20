import os
from ...models.datasource_specs.adls import ADLSDataSourceSpec, AuthSecretSpec

from dotenv import load_dotenv
load_dotenv()

adls_data_source_spec = ADLSDataSourceSpec(
    storage_account_name=os.environ["CFPA_STORAGE_ACCOUNT_NAME"],
    container_name="corpfpa",
    auth_secret=AuthSecretSpec(
        secret_suffix="AIHUB_INDEXING",
    )
)