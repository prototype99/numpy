
def NotImplementedFunc(func):
    def tmp(*args, **kwargs):
        raise NotImplementedError("%s not implemented yet" % func)
    return tmp

for name in '''
eigvals eig eigh_lo cholesky_lo svd_n_f svd_m inv
'''.split():
    if name not in globals():
        globals()[name] = NotImplementedFunc(name)

