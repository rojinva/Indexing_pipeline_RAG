from ._base_metrics import RetrievalMetric


class RecallAtK(RetrievalMetric):
    """
    Metric to calculate recall at different cutoffs k.
    """

    def __init__(self, k, compute_all_k=False):
        """
        Initializes the RecallAtK metric.

        Parameters:
            k (int): The cutoff k at which recall is calculated.
            compute_all_k (bool): Whether to compute recall at all values from 1 to k.
        """
        super().__init__()
        self.k = k
        self.compute_all_k = compute_all_k

    def calculate(self, ground_truths, retrievals):
        """
        Calculates the recall at k or at all k up to the specified cutoff.

        Parameters:
            ground_truths (list): The list of ground truth documents.
            retrievals (list): The list of retrieved documents up to k.

        Returns:
            dict: A dictionary with keys as 'Recall@i' and boolean values indicating recall success.
        """
        results = {}
        if self.compute_all_k:
            for i in range(1, self.k + 1):
                results[f"Recall@{i}"] = set(ground_truths).issubset(
                    set(retrievals[:i])
                )
        else:
            results[f"Recall@{self.k}"] = set(ground_truths).issubset(
                set(retrievals[: self.k])
            )
        return results
