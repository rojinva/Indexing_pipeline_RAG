import os
import json
import argparse
import asyncio
import openpyxl
import pandas as pd

# Append directory to system path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from dotenv import load_dotenv

load_dotenv(override=True)

from datetime import datetime
from src.core.retrievers import AzureAISearchRetriever
from src.core.metrics import RecallAtK
from src.core.evaluators import RetrievalEvaluator


async def grid_search(evaluator, search_configs):
    """
    Perform a grid search over specified search configurations using the provided evaluator and append results to the DataFrame.
    Parameters:
        evaluator (RetrievalEvaluator): An evaluator that encapsulates the data, metrics, and retrieval logic.
        search_configs (list): List of dictionaries, each representing a search configuration.
    Returns:
        pd.DataFrame: Original DataFrame augmented with the results for each configuration.
    """
    all_results = []
    for config in search_configs:
        search_type, search_params = next(iter(config.items()))
        print(f"Running Grid Search for {search_type} config.")
        await evaluator.evaluate(os.environ["RETRIEVAL_COLS"].split(","), search_params)
        for row in evaluator.evaluated_dataframe.iterrows():
            result_data = {
                "config": search_type,
                "config_params": search_params,
            }
            all_results.append({**row[1].to_dict(), **result_data})
    return pd.DataFrame(all_results)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run grid search to optimize document retrieval."
    )
    parser.add_argument(
        "dataframe_path",
        type=str,
        help="Path to the CSV file containing the DataFrame.",
    )
    parser.add_argument(
        "search_params_path",
        type=str,
        help="Path to the JSON file containing the search parameters.",
    )
    return parser.parse_args()


async def run_grid_search(dataframe_path, search_params_path):
    """Asynchronously run the grid search using the specified DataFrame and search parameters."""
    # Load the DataFrame
    workbook = openpyxl.load_workbook(dataframe_path)
    sheet = workbook["Sheet1"]
    data = list(sheet.values)[1:]  # Skip non-header rows
    df = pd.DataFrame(data, columns=list(sheet.values)[0])  # Convert to DataFrame
    # Load search configurations
    with open(search_params_path, "r") as f:
        search_configs = json.load(f)
    # Setup the Azure AI search retriever and metrics
    search_service = AzureAISearchRetriever(
        search_api_base=os.environ["AZURE_AI_SEARCH_BASE"],
        index_name=os.environ["AZURE_AI_SEARCH_INDEX_NAME"],
        search_api_version=os.environ["AZURE_AI_SEARCH_API_VERSION"],
        api_key=os.environ["AZURE_AI_SEARCH_API_KEY"],
    )

    metrics = [RecallAtK(k=int(os.environ["RECALL_AT_K"]), compute_all_k=True)]
    evaluator = RetrievalEvaluator(
        dataframe=df,
        ground_truth_cols=os.environ["GROUND_TRUTH_COLS"].split(","),
        search_service=search_service,
        metrics=metrics,
    )
    # Execute grid search
    results_df = await grid_search(evaluator, search_configs)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # results_df.to_csv(f'grid_search_results_{timestamp}.csv', index=False)
    results_df.to_csv("src\scripts\grid_search\grid_search_results.csv", index=False)


def main():
    args = parse_args()
    asyncio.run(run_grid_search(args.dataframe_path, args.search_params_path))


if __name__ == "__main__":
    main()
