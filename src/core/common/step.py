from abc import ABC


class Step(ABC):
    """
    Abstract base class for all steps in the evaluation pipeline.
    """

    def __init__(self):
        """
        Initializes the Step with its name set to the class name.
        """
        self.name = self.__class__.__name__
