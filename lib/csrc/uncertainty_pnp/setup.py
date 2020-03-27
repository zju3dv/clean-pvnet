import os

ceres_include='./include'
ceres_library='./lib/libceres.so'
eigen_include='./include/eigen3'
glog_library='./lib/libglog.so'
os.system('gcc -shared src/uncertainty_pnp.cpp -c -o src/uncertainty_pnp.cpp.o -fopenmp -fPIC -O2 -std=c++11 -I {} -I {}'.format(ceres_include,eigen_include))

from cffi import FFI
ffibuilder = FFI()


with open(os.path.join(os.path.dirname(__file__), "src/ext.h")) as f:
    ffibuilder.cdef(f.read())

ffibuilder.set_source(
    "_ext",
    """
    #include "src/ext.h"
    """,
    extra_objects=['src/uncertainty_pnp.cpp.o',
                   ceres_library,
                   glog_library],
    libraries=['stdc++']
)


if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
    os.system("rm src/*.o")
    os.system("rm *.o")
