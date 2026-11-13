import numpy as np
from numba import njit


@njit(cache=True)
def _integrate(y, valid, bins, counts, starts, ends, median_window):
    n = len(y)
    positive = 0
    n_valid = 0
    for j in range(n):
        if valid[j]:
            n_valid += 1
            positive += y[j]
    mean_roc = 0.0
    mean_pr = 0.0
    range_roc = 0.0
    range_pr = 0.0
    max_window = 2 * median_window
    for window in range(max_window + 1):
        weights = np.zeros(250)
        hits = np.zeros(250)
        mass = 0.0
        segments = 0
        minimum = 250
        half = window // 2
        for j in range(n):
            value = float(y[j])
            if value == 0.0 and window > 0:
                for k in range(len(starts)):
                    if j < starts[k] and starts[k] - j <= half:
                        value += np.sqrt(1.0 - (starts[k] - j) / window)
                    if j > ends[k] and j - ends[k] < half:
                        value += np.sqrt(1.0 - (j - ends[k]) / window)
                value = min(value, 1.0)
            if valid[j]:
                mass += value
                weights[bins[j]] += value

            if value > 0.0:
                if valid[j]:
                    minimum = min(minimum, bins[j])
            elif minimum < 250:
                hits[minimum] += 1
                segments += 1
                minimum = 250
        if minimum < 250:
            hits[minimum] += 1
            segments += 1
        pnew = (positive + mass) / 2.0
        nnew = n_valid - pnew
        tp = 0.0
        existence = 0.0
        last_tpr = 0.0
        last_fpr = 0.0
        last_prec = 1.0
        roc = 0.0
        pr = 0.0
        for k in range(250):
            tp += weights[k]
            existence += hits[k]
            tpr = min(tp / pnew, 1.0) * existence / segments
            fpr = (counts[k] - tp) / nnew
            precision = tp / counts[k]
            roc += (fpr - last_fpr) * (tpr + last_tpr) / 2.0
            pr += (tpr - last_tpr) * (precision + last_prec) / 2.0
            last_tpr = tpr
            last_fpr = fpr
            last_prec = precision
        roc += (1.0 - last_fpr) * (1.0 + last_tpr) / 2.0
        mean_roc += roc
        mean_pr += pr
        if window == median_window:
            range_roc = roc
            range_pr = pr
    return range_roc, range_pr, mean_roc / (max_window + 1), mean_pr / (max_window + 1)


def range_vus(y, scores, valid):
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    n_valid = int(valid.sum())
    positive = int(y[valid].sum())
    if n_valid == 0 or positive == 0:
        return np.nan, np.nan, np.nan, np.nan
    if positive == n_valid:
        return np.nan, 1.0, np.nan, 1.0
    diff = np.diff(np.r_[0, y, 0])
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    window = max(1, int(np.median(ends - starts + 1)))
    thresholds = -np.sort(-scores[valid])[np.linspace(0, n_valid - 1, 250).astype(int)]
    bins = np.searchsorted(-thresholds, -scores, side="left")
    counts = np.cumsum(np.bincount(bins[valid], minlength=250))
    return _integrate(y, valid, bins, counts, starts, ends, window)
