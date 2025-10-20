from ..common.step import Step


class BaseEvaluator(Step):
    """
    Abstract base class for all evaluators.
    """

    def __init__(self):
        """
        Initializes the BaseEvaluator, inheriting properties from Step.
        """
        super().__init__()

    async def evaluate(self):
        """
        Abstract method to evaluate.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Evaluate method is missing...")
