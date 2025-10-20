# Introduction 
This repository marks a strategic pivot in our approach to indexing and search capabilities through Azure AI Search, transitioning from a push model to a pull model. In a push model, data must be actively sent to the search index by the application, often resulting in inefficiencies and stale data. The pull model, by contrast, allows Azure AI Search to directly pull data from the source whenever updates are detected, ensuring more current data and reducing system load. This model not only streamlines data synchronization but also enhances the overall efficiency and scalability of our search operations.

# Why the Pull Model?
We chose the pull model for its automatic data synchronization capabilities, which minimize manual intervention and improve data accuracy and timeliness. This approach is particularly beneficial in environments where data changes frequently and needs to be indexed in near real-time.

## Change Detection and Delete Detection
1. The indexers are equipped with Change Detection capabilities, which track modifications in the data source to only index new or updated items. This feature ensures efficient use of resources by avoiding the re-indexing of unchanged data.
2. The Delete Detection Policy uses custom metadata of the blob to manage deletions. Specifically, if `is_deleted` is marked as `True` in the blob's metadata, the indexer recognizes this signal and removes the corresponding chunks of the file from the index during its next run, maintaining the integrity and relevance of the search index.

# Getting Started

## Azure Function App Setup and Testing

# Local Development
Navigate to the Azure Function project directory:

```bash
cd src/search_service/skills
```
Start the function app locally:
```bash
func host start
```

# Setting Up Postman for Local Testing
1. Open Postman and create a `New HTTP Request`.
2. Set the method to POST and input the function's URL noted from the local start.
3. Configure necessary headers: 
    - `size` (chunk size)
    - `overlap` (chunk overlap)
3. Use this sample JSON body required by our Custom Split Skill. 
```json
{
    "values": [
        {
            "recordId": "1",
            "data": {
                "content": "What is Lam's Response to Everything?",
                "parent_filename": "sample_file.xlsx",
                "blob_uri": "https://sample.blob.core.windows.net/sample_container/sample_folder/sample_file.xlsx"
            }
        }
    ]
}

```
4. Send the request and analyze the response.

## CI/CD Practices
Automated pipelines manage deployment and ensure system reliability and performance. 

### Pipeline Triggers

1. **Configuration Changes**: Any commits affecting `src/search_service` configuration files automatically trigger updates across environments (based on the branch. For eg. A change in dev branch would trigger updates to the dev environment.)
2. **Skill Updates**: Modifications to `src/search_service/skillset/skills` initiates an automatic redeployment to the Azure Function App.


## Software Dependencies
1. "excel_splitter.py" uses a connection to Azure blob storage. To establish this connection, the code uses  tenant id, client id, client secret and 
storage account name. These values are retrieved from a key vault. The client id used to access the storage account *must* have Reader access to the
storage container. This was missing by default. 


## Running Tests

To run the tests, navigate to the project root directory and execute the following command:

```python
python -m unittest discover tests
```
This command will discover and run all tests in the tests directory, reporting any failures and summarizing the test results.

### Writing New Tests

When contributing new features or fixes, please include corresponding unit tests. Tests should be concise, isolated, and targeted, focusing on small units of functionality. Follow existing patterns in the codebase for consistency.

By maintaining a robust suite of unit tests, we ensure the application remains reliable and bug-free over time.

# Contribute

Contributing to the project through Azure DevOps encompasses more than just code. It involves keeping the project documentation accurate and the development process transparent. Here’s how you can contribute effectively:

## Reporting Issues

Use Azure DevOps to file work items for bugs or feature requests. Include detailed reproduction steps, expected outcomes, and relevant screenshots.

## Submitting Changes

1. **Access the Repository**: Ensure you have access to our Azure DevOps repository. If not, request permissions from the project administrators.
2. **Fork and Clone**: Fork the repository within Azure DevOps and clone it to your local environment.
3. **Create a Branch**: Create a new branch for your specific changes. This helps isolate feature or bug fixes from the main development line.
4. **Implement Changes**: Develop your feature or fix within your branch. Ensure your changes adhere to existing code standards and practices.
5. **Write Tests**: Add or update unit tests to cover new or changed functionality. Ensure all tests pass to maintain code quality.
6. **Update the Changelog**:
    - Navigate to the `src` directory and update the `CHANGELOG.md`.
    - Classify your changes under `Added`, `Changed`, `Fixed`, or `Removed`.
    - Provide a clear description of what was done and reference the work item numbers related to your changes.
    - Maintain clarity and consistency in your entries to ensure they are useful and understandable.
    - **Push Changes**: Commit your changes and push the branch back to Azure DevOps.
    - **Create a Pull Request**: Initiate a pull request against the main branch. Link your pull request to any related work items to provide context and facilitate tracking.

## Code Reviews

Once your pull request is up, it will undergo a review process by project maintainers. Engage constructively with any feedback to refine your contribution.

## Stay Updated

Regularly pull changes from the main repository to your fork to keep it current. This practice helps reduce merge conflicts and simplifies the integration of your contributions.

By actively participating in these processes, you contribute to the robustness and success of the project.

## Python Environment Setup

### Prerequisites

This project requires **Python 3.11.13** to work correctly with Azure Functions and the latest security updates.

### Check Existing Python Installation

First, check if Python 3.11.13 is already available on your system:

```powershell
py --list
```

Look for `CPython 3.11.13` in the output. If you see it, skip to the [Setting Up the Virtual Environment](#setting-up-the-virtual-environment) section.

### Installing Python 3.11.13

If Python 3.11.13 is not installed, choose one of the following methods:

> **Note:** Python 3.11.13 is a security patch release and does not have a standalone binary installer from python.org. Use UV or Poetry for installation.

#### Option 1: Using UV (Recommended - Fast and Lightweight)

UV is developed by Astral and provides fast Python version management:

```powershell
# Install UV
iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex

# Install Python 3.11.13
uv python install 3.11.13

# Verify installation - you should see CPython 3.11.13 listed
py --list
```

**Note:** When installed via UV, Python will appear as `Astral/CPython3.11.13` in the output.

#### Option 2: Using Poetry

Poetry can manage Python versions for project-specific environments:

```powershell
# Install Poetry
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Configure Poetry to use Python 3.11.13 (requires Python 3.11.13 to be installed first)
poetry env use 3.11.13
```

### Setting Up the Virtual Environment

Once Python 3.11.13 is installed, follow these steps to set up your development environment:

1. **Remove existing virtual environment** (if it exists):
   ```powershell
   Remove-Item -Recurse -Force .venv
   ```

2. **Create a new virtual environment with Python 3.11.13**:
   
   Find the exact Python version identifier from `py --list`, then use it to create the virtual environment:
   
   ```powershell
   # If installed via UV (will show as Astral/CPython3.11.13)
   py -V:Astral/CPython3.11.13 -m venv .venv
   
   # If installed from python.org or other sources (will show as 3.11)
   py -3.11 -m venv .venv
   ```

3. **Activate the virtual environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. **Verify Python version**:
   ```powershell
   python --version
   ```
   
   Expected output: `Python 3.11.13`

5. **Install project dependencies**:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r src\search_service\skills\requirements.txt
   ```

### Troubleshooting

- **Permission Denied Error**: If you encounter a permission denied error when removing `.venv`, make sure to:
  - Run `deactivate` in any terminal with an active virtual environment
  - Close all terminals that might be using the virtual environment
  - Close VS Code and reopen it if necessary
  
- **Python Version Not Found**: Run `py --list` to see the exact identifier for your Python 3.11.13 installation and use that exact string with the `-V:` flag.

- **Execution Policy Error**: If you cannot run PowerShell scripts, adjust your execution policy:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

- **UV Not Found After Installation**: Close and reopen your PowerShell terminal to refresh the PATH environment variable.




"# Indexing_pipeline_RAG" 
