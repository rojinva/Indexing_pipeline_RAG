import asyncio

import pandas as pd
from ._base_evaluators import BaseEvaluator
from tqdm.asyncio import tqdm


class RetrievalEvaluator(BaseEvaluator):
    """
    Evaluator for retrieval tasks using an Azure AI search retriever with rate limiting.
    """

    def __init__(self, dataframe, ground_truth_cols, search_service, metrics):
        """
        Initializes the RetrievalEvaluator.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the data to be evaluated.
            ground_truth_cols (list): Column names in df that contain the ground truth data.
            search_service (AzureAISearchRetriever): The search service used for retrieving documents.
            metrics (list): List of Metric instances to be calculated.
        """
        super().__init__()
        self.dataframe = dataframe
        self.ground_truth_cols = ground_truth_cols
        self.search_service = search_service
        self.metrics = metrics
        self.evaluated_dataframe = None

    async def evaluate(self, fields_to_retrieve_from_index, search_params={}):
        """
        Asynchronously evaluates the retrieval performance using the specified metrics.

        Parameters:
            search_params (dict): If empty dictionary, retriever uses default search params.

        Returns:
            list: List of individual results.
        """

        # Initialize a progress bar with the total number of tasks
        progress_bar = tqdm(total=len(self.dataframe), desc="Evaluating Questions")
        results_list = []
        # Loop through each row, process it, and update the progress bar
        for _, row in self.dataframe.iterrows():
            result = await self._evaluate_row(
                row["Question"], fields_to_retrieve_from_index, row, search_params
            )
            results_list.append(result)
            progress_bar.update(1)  # Update the progress bar after each completion
        progress_bar.close()
        self.evaluated_dataframe = self._append_results_to_dataframe(results_list)
        return results_list

    async def _evaluate_row(
        self, question, fields_to_retrieve_from_index, row, search_params
    ):
        """
        Evaluates a single row from the DataFrame.

        Parameters:
            question (str): The question to retrieve documents for.
            fields (list): List of fields to retrieve for each document.
            row (pd.Series): A single row from the DataFrame representing one evaluation case.

        Returns:
            dict: A dictionary containing the calculated metric results for the row.
        """
        k = max([metric.k for metric in self.metrics])
        search_params["top"] = k
        retrievals = [
            "_".join(retrieval)
            for retrieval in await self.search_service.retrieve(
                question, fields_to_retrieve_from_index, search_params
            )
        ]
        ground_truths = ["_".join([row[col] for col in self.ground_truth_cols])]
        result_dict = {"retrievals": retrievals}
        for metric in self.metrics:
            metric_results = metric.calculate(ground_truths, retrievals)
            result_dict.update(metric_results)
        return result_dict

    def _append_results_to_dataframe(self, results_list):
        """
        Attaches individual evaluation results and retrieval results to the original DataFrame,
        splitting the retrievals into separate columns dynamically for each retrieved document.
        Parameters:
            results_list (list of dict): List containing dictionaries with recall results and retrieval data for each query.
        Returns:
            pd.DataFrame: DataFrame with the results attached, formatted with dynamic retrieval columns.
        """
        # Convert results list to a DataFrame
        results_df = pd.DataFrame(results_list)
        # Use apply to transform the 'retrievals' list into separate columns
        if "retrievals" in results_df.columns:
            retrievals_df = results_df["retrievals"].apply(pd.Series)
            retrievals_df.columns = [
                f"retrieval_{i+1}" for i in retrievals_df.columns
            ]  # Rename columns to 'retrieval_i'
            # Drop the original 'retrievals' column from results_df
            results_df = results_df.drop(columns="retrievals")
            # Concatenate the expanded retrievals DataFrame with the results DataFrame
            results_df = pd.concat([results_df, retrievals_df], axis=1)
        # Ensure the index aligns if the DataFrame and results list are not in order
        results_df.index = self.dataframe.index
        # Concatenate the original DataFrame and the modified results DataFrame
        combined_df = pd.concat([self.dataframe, results_df], axis=1)
        return combined_df
