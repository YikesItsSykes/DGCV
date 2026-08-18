def _spool(multi_index, shape):
    idx = 0
    stride = 1
    for i, s in zip(reversed(multi_index), reversed(shape)):
        idx += i * stride
        stride *= s
    return idx


def _unspool(index, shape):
    multi = []
    for s in reversed(shape):
        multi.append(index % s)
        index //= s
    return tuple(reversed(multi))
