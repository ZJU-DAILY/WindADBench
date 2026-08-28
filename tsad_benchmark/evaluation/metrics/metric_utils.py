
import numpy as np


def get_list_anomaly(labels: np.ndarray) -> np.ndarray:

    end_pos = np.diff(np.asarray(labels, dtype=int), append=0) < 0
    return np.diff(np.cumsum(labels)[end_pos], prepend=0)
