from ..common.step import Step


class BaseMetric(Step):
    """
    Abstract base class for all metrics that can be calculated.
    """

    def __init__(self):
        """
        Initializes the Metric, inheriting properties from Step.
        """
        super().__init__()

    def calculate(self):
        """
        Abstract method to calculate the metric.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Calculate method is missing...")


class RetrievalMetric(BaseMetric):
    """
    Abstract base class for metrics specific to retrieval tasks.
    """

    def __init__(self):
        """
        Initializes the RetrievalMetric, inheriting properties from Metric.
        """
        super().__init__()

    def calculate(self):
        """
        Abstract method to calculate the retrieval metric.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Retrieval Calculate method is missing...")
