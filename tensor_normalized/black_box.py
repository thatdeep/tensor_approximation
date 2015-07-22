import numpy as np

class BlackBox(object):
    def __init__(self, f, f_vect, bounds, n, d, dtype=np.float, array_based=False):
        self.space = np.linspace(bounds[0], bounds[1], n, endpoint=False)
        self.bounds = bounds
        self.array_based = array_based
        if not hasattr(n, "__len__"):
            self.n = tuple([n]*d)
        else:
            self.n = n
        self.shape = self.n
        self.f = f
        self.f_vect = f_vect
        self.d = d
        self.dtype=dtype

    @classmethod
    def from_array(cls, data):
        if isinstance(data, np.ndarray):
            return cls(lambda i: data[i], lambda ii: [data[i] for i in ii], data.shape, len(data.shape), data.dtype, array_based=True)

    def pp(self, item):
        return np.asarray(item)

    def __getitem__(self, item):
        if isinstance(item, np.ndarray) and len(item.shape) == 2:
            vals = self.space[item.ravel()].reshape(item.shape)
            return self.f_vect(vals)
        it = self.pp(item)
        if it.shape == (self.d, ):
            return self.f(self.space[it])
        elif it.shape and len(it.shape) == 2 and it.shape[0] == self.d:
            if self.array_based:
                return self.f(it)
            else:
                vals = self.space[it.ravel()].reshape(it.shape)
                return self.f_vect(vals)
                return np.apply_along_axis(self.f, 1, vals.T)
                #return np.fromiter((self.f(col) for col in it.T), dtype=self.dtype)