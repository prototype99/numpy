Welcome to the ChimOS fork of PyPy's fork of Numpy, a fundamental package needed for scientific computing with Python *NumPyPy*. In order to install, first
install PyPy, hints are here http://pypy.org/download.html. Note this is
a binary install, no lengthy translation or compilation necessary.

If you get a message about `missing Python.h` you must install the pypy-dev
package for your system

If you get a message about "unable to find vcvarsall.bat", you need to install
install a compiler. Microsoft has a download for that at
http://www.microsoft.com/en-us/download/details.aspx?id=44266

If you installed to a system directory, you may need to run::

    sudo pypy -c 'import numpy'

once to initialize the cffi cached shared objects as `root`

numpy is still not pypy3 compatible

If you do not have lapack/blas runtimes, it may take over 10 minutes to install,
since it needs to build a lapack compatability library. However, you may later
install upstream compatible runtimes, and NumPy should pick them up
automatically the next time you run PyPy.

Also note that the latest version of NumPy will probably not run in an older
PyPy. Specifically, we require cffi 1.0 or later. Since cffi is baked into
PyPy, you cannot update cffi in any version of PyPy (true as of Nov 2015)
so there is no recourse but to update PyPy.


- **Website:** https://www.numpy.org
- **Documentation:** https://numpy.org/doc
- **Mailing list:** https://mail.python.org/mailman/listinfo/numpy-discussion
- **Source code:** https://github.com/numpy/numpy
- **Contributing:** https://www.numpy.org/devdocs/dev/index.html
- **Bug reports:** https://github.com/numpy/numpy/issues
- **Report a security vulnerability:** https://tidelift.com/docs/security

It provides:

- a powerful N-dimensional array object
- sophisticated (broadcasting) functions
- tools for integrating C/C++ and Fortran code
- useful linear algebra, Fourier transform, and random number capabilities

It derives from the old Numeric code base and can be used as a replacement for Numeric. It also adds the features introduced by numarray and can be used to replace numarray.

Testing:

After installation, tests can be run with:

    python -c 'import numpy; numpy.test()'