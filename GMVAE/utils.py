from enum import Enum

class LatentType(Enum):
    GAUSSIAN = 1
    DIRICHLET = 2
    CONTINUOUS_CATEGORICAL = 3