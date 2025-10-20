from setuptools import find_packages, setup
from src._version import __version__

setup(
    name="ent-openai-search-index",
    version=__version__,
    description="",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["requests", "azure-search-documents==11.6.0b1", "python-dotenv"],
    extras_require={
        "dev": ["black"],
    },
)
