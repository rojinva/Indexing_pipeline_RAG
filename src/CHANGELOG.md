# Change Log
All notable changes to this project will be documented in this file.

## Released (April 18, 2024) (v2.0)

### Added
- Supported Usecases: Customer Survey, Common Spec
- ADLS Indexer based approach for processing documents in ADLS
- Function App with Custom Split Skill
- DevOps Pipelines to deploy function app, AIS resources for usecases


## Unreleased
- added configuration for quest-iplm


### Added
- added techMemo skill set definition
- Added Vector Embedding Skill to project vectors into the index
- Onboarding New Usecases: Iplm, EngineeringELT, techMemo
- Document Intelligence Custom Preprocessing to process pdf, ppt, word - Iplm, EngineeringElt
- Added Core Library (includes Retrievers, Metrics, Evaluators)
- Added Scripts (grid search over benchmark dataset and search params)
- Add Readme documentation for Repository and Core Library
- Added iplm skill set definition
- Added configurations for LIGHTCAE CHATBOT project

### Changed
- Updated semaphore to 5 concurrent requests
- Modified data cleaning process to drop only empty rows, retaining all columns for Customer Survey

### Fixed
- Updated retry decorator to also capture the actual error in case of exceptions

### Removed