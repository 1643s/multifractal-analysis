import numpy as np
def Event(series):
    if len(series) == 0:
        return []
    result = [series[0]]
    for x in series[1:]:
        if x != result[-1]:
            result.append(x)
    return result