class DatasetWrapper(object):
    
    def __init__(self, ds, n_examples):
        self.batch_size = 1
        self.n_examples = n_examples
        self.length = n_examples
        self.ds = ds


    def map(self, *args, **kwargs):
        self.ds = self.ds.map(*args, **kwargs)
        return self


    def batch(self, size):
        self.ds = self.ds.batch(size)
        self.batch_size = size
        self.length = int(self.n_examples / size)
        if self.n_examples % size is not 0:
            self.length += 1
        return self


    def prefetch(self, arg):
        self.ds = self.ds.prefetch(arg)
        return self


    def shuffle(self, buf_size):
        self.ds = self.ds.shuffle(buf_size)
        return self


    def __len__(self):
        return self.length


    def __iter__(self):
        return iter(self.ds)