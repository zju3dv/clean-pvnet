import os

# os.system('gcc -shared src/farthest_point_sampling.cpp -c -o src/farthest_point_sampling.cpp.o -fopenmp -fPIC -O2 -std=c++11')
os.system('cl.exe src/farthest_point_sampling.cpp /c /Fo"src/farthest_point_sampling.obj" /MD /EHsc /openmp /O2 /std:c++14')

from cffi import FFI
ffibuilder = FFI()


with open(os.path.join(os.path.dirname(__file__), "src/ext.h")) as f:
    ffibuilder.cdef(f.read())

ffibuilder.set_source(
    "_ext",
    """
    #include "src/ext.h"
    """,
    extra_objects=['src/farthest_point_sampling.obj'],
    # libraries=['stdc++']
)


if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
    os.system("del /q src\*.obj")
    os.system("del /q *.obj")
