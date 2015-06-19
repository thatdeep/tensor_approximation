import numpy as np

class BlackBox(object):
    def __init__(self, f, bounds, n, d, dtype=np.float, array_based=False):
        self.array_based = array_based
        if not hasattr(n, "__len__"):
            self.n = tuple([n]*d)
        else:
            self.n = n
        self.shape = self.n
        self.f = f
        self.d = d
        self.dtype=dtype
        self.spaces =

    @classmethod
    def from_array(cls, data):
        if isinstance(data, np.ndarray):
            return cls(lambda i: data[i], data.shape, len(data.shape), data.dtype, array_based=True)

    def __getitem__(self, item):
        it = np.asarray(item)
        if it.shape == (self.d, ):
            return self.f(it)
        elif it.shape and len(it.shape) == 2 and it.shape[0] == self.d:
            if self.array_based:
                return self.f(it)
            else:
                return np.apply_along_axis(self.f, 1, it.T)
                #return np.fromiter((self.f(col) for col in it.T), dtype=self.dtype)