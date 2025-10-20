from ..common.step import Step


class BaseRetriever(Step):
    """
    Abstract base class for all retrievers.
    """

    def __init__(self):
        """
        Initializes the BaseRetriever, inheriting properties from Step.
        """
        super().__init__()

    async def retrieve(self):
        """
        Abstract method to retrieve.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Retrieve method is missing...")
